/**
 * @file   TaskViapoint.cpp
 * @brief  TaskViapoint class source file.
 * @author Freek Stulp
 *
 * This file is part of DmpBbo, a set of libraries and programs for the
 * black-box optimization of dynamical movement primitives.
 * Copyright (C) 2014 Freek Stulp, ENSTA-ParisTech
 *
 * DmpBbo is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 2 of the License, or
 * (at your option) any later version.
 *
 * DmpBbo is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with DmpBbo.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <boost/serialization/export.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include "/home/hrcstudents/Documents/Code/catkin_ws/src/dmp_bbo_node/include/dmp_bbo_node/TaskViapoint.hpp"

/** For boost::serialization. See http://www.boost.org/doc/libs/1_55_0/libs/serialization/doc/special.html#export */
BOOST_CLASS_EXPORT_IMPLEMENT(DmpBbo::TaskViapoint);

#include <boost/serialization/base_object.hpp>

#include <iostream>
#include <fstream>
#include <iomanip>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>


#include "/home/hrcstudents/Documents/Code/catkin_ws/src/dmp_bbo_node/include/dmp_bbo_node/EigenBoostSerialization.hpp"
#include "/home/hrcstudents/Documents/Code/catkin_ws/src/dmp_bbo_node/include/dmp_bbo_node/BoostSerializationToString.hpp"

#include "/home/hrcstudents/Documents/Code/catkin_ws/src/dmp_bbo_node/include/dmp_bbo_node/EigenFileIO.hpp"

#include <vector>
#include <map>
#include <limits>
#include <cmath>
#include <cstdio>
//#include <boost/tuple/tuple.hpp>
//#include <boost/foreach.hpp>
#include "/home/hrcstudents/Documents/Code/catkin_ws/src/dmp_bbo_node/include/dmp_bbo_node/gnuplot-iostream.h"
 #include "ros/ros.h"

 #include "std_msgs/String.h"
#include "geometry_msgs/Point.h"
 #include <sstream>
 #include <unistd.h>

using namespace std;
using namespace Eigen;

namespace DmpBbo {

TaskViapoint::TaskViapoint(const Eigen::VectorXd& viapoint, double  viapoint_time, double viapoint_radius)
: viapoint_(viapoint), viapoint_time_(viapoint_time), viapoint_radius_(viapoint_radius),
  goal_(VectorXd::Ones(viapoint.size())), goal_time_(-1),
  viapoint_weight_(1.0), acceleration_weight_(0.0001),  goal_weight_(0.0),

goal_one_(viapoint), goal_two_(viapoint), goal_selected_(1),

accuracy_weight_(20), robot_time_weight_(1.0),  human_time_weight_(5), jerk_weight_(50)
{
  assert(viapoint_radius_>=0.0);
}

TaskViapoint::TaskViapoint(const Eigen::VectorXd& viapoint, double  viapoint_time, const Eigen::VectorXd& goal,  double goal_time)
: viapoint_(viapoint), viapoint_time_(viapoint_time), viapoint_radius_(0.0),
  goal_(goal), goal_time_(goal_time),

  viapoint_weight_(1.0), acceleration_weight_(0.0001),  goal_weight_(1.0),

  goal_one_(viapoint), goal_two_(viapoint), goal_selected_(1),

  accuracy_weight_(20), robot_time_weight_(1.0),  human_time_weight_(5),jerk_weight_(50)
{
  assert(viapoint_.size()==goal.size());
}

//New Constructor
TaskViapoint::TaskViapoint(const Eigen::VectorXd& goal_one, const Eigen::VectorXd& goal_two, const Eigen::VectorXd& goal,  double goal_selected, double goal_time)
:
  viapoint_(goal_one), viapoint_time_(0.5), viapoint_radius_(2),

 goal_(goal), goal_time_(goal_time),
    viapoint_weight_(1.0), acceleration_weight_(1),  goal_weight_(0.0),

  goal_one_(goal_one), goal_two_(goal_two), goal_selected_(goal_selected),

  accuracy_weight_(3000), robot_time_weight_(150),  human_time_weight_(1000), jerk_weight_(50)
{
  assert(goal_one_.size()==goal_two_.size());
}

void TaskViapoint::evaluate(const MatrixXd& cost_vars, const MatrixXd& task_parameters, VectorXd& costs) const
{
  int n_samples = cost_vars.rows();
  int n_dims = viapoint_.size();
  int n_cost_vars = 4*n_dims + 1; // y_1..y_D  yd_1..yd_D  ydd_1..ydd_D  t forcing_term_1..forcing_term_D
  int n_time_steps = cost_vars.cols()/n_cost_vars;
  // cost_vars  = n_samples x (n_time_steps*n_cost_vars)

  //cout << "  n_samples=" << n_samples << endl;
  //cout << "  n_dims=" << n_dims << endl;
  //cout << "  n_cost_vars=" << n_cost_vars << endl;
  //cout << "  cost_vars.cols()=" << cost_vars.cols() << endl;
  //cout << "  n_time_steps=" << n_time_steps << endl;


  costs.resize(n_samples);

  // y_1..y_D  yd_1..yd_D  ydd_1..ydd_D  t=0
  // y_1..y_D  yd_1..yd_D  ydd_1..ydd_D  t=1
  //  |    |     |     |      |     |     |
  // y_1..y_D  yd_1..yd_D  ydd_1..ydd_D  t=T
  MatrixXd rollout; //(n_time_steps,n_cost_vars);
  MatrixXd my_row(1,n_time_steps*n_cost_vars);

  MatrixXd rollout2; //(n_time_steps,n_cost_vars);
  MatrixXd my_row2(1,n_time_steps*n_cost_vars);
  for (int k=0; k<n_samples; k++)
  {
    my_row = cost_vars.row(k);
    rollout = (Map<MatrixXd>(my_row.data(),n_cost_vars,n_time_steps)).transpose();

    // rollout is of size   n_time_steps x n_cost_vars
    VectorXd ts = rollout.col(3 * n_dims);


    double dist_to_viapoint = 0.0;
    if (viapoint_weight_!=0.0)
    {
      if (viapoint_time_ == TIME_AT_MINIMUM_DIST)
      {
        // Don't compute the distance at some time, but rather get the minimum distance
        const MatrixXd y = rollout.block(0, 0, rollout.rows(), n_dims);
        dist_to_viapoint = (y.rowwise() - viapoint_.transpose()).rowwise().squaredNorm().minCoeff();
      }
      else
      {
        // Compute the minimum distance at a specific time

        // Get the time_step at which viapoint_time_step approx ts[time_step]
        int viapoint_time_step = 0;
        while (viapoint_time_step < ts.size() && ts[viapoint_time_step] < viapoint_time_)
          viapoint_time_step++;

        assert(viapoint_time_step < ts.size());

        VectorXd y_via = rollout.row(viapoint_time_step).segment(0,n_dims);
        dist_to_viapoint = sqrt((y_via-viapoint_).array().pow(2).sum());
      }

      if (viapoint_radius_>0.0)
      {
        // The viapoint_radius defines a radius within which the cost is always 0
        dist_to_viapoint -= viapoint_radius_;
        if (dist_to_viapoint<0.0)
          dist_to_viapoint = 0.0;
      }
    }

    double sum_ydd = 0.0;
    MatrixXd ydd;
    ydd = rollout.block(0,2*n_dims,n_time_steps,n_dims);

    if (acceleration_weight_!=0.0)
    {
     // ydd = rollout.block(0,2*n_dims,n_time_steps,n_dims);
      // ydd = n_time_steps x n_dims
      sum_ydd = ydd.array().pow(2).sum();
    }



    double delay_cost = 0.0;
    if (goal_weight_!=0.0)
    {
      int goal_time_step = 0;
      while (goal_time_step < ts.size() && ts[goal_time_step] < goal_time_)
        goal_time_step++;

      const MatrixXd y_after_goal = rollout.block(goal_time_step, 0,
        rollout.rows() - goal_time_step, n_dims);

      delay_cost = (y_after_goal.rowwise() - goal_.transpose()).rowwise().squaredNorm().sum();
    }

    costs[k] =
      viapoint_weight_*dist_to_viapoint +
      acceleration_weight_*sum_ydd/n_time_steps +
      goal_weight_*delay_cost;
  }
}

/*void TaskViapoint::getFeedback(double human_time,double accuracy,ros::Time begin,ros::Subscriber key,int a)
{ros::Time end=ros::Time::now();
human_time=end-begin;
if(a==1){
if(key.code==SDL_KEYUP)
accuracy=0;
else accuracy=1;}}*/

void TaskViapoint::evaluateHrc(const MatrixXd& cost_vars1,const MatrixXd& cost_vars2, const MatrixXd& task_parameters, VectorXd& costs1,VectorXd& costs2, int update,ros::NodeHandle node,ros::Publisher traj_pub) const
{
  int n_samples = cost_vars1.rows();
  int n_dims = goal_one_.size();
  int n_cost_vars = 4*n_dims + 1; // y_1..y_D  yd_1..yd_D  ydd_1..ydd_D  t forcing_term_1..forcing_term_D
  int n_time_steps = cost_vars1.cols()/n_cost_vars;
  double time_weight=2.5;
  //double time_weight=1.82*goal_time_; //goal_time_ = tau
  // cost_vars  = n_samples x (n_time_steps*n_cost_vars)

  //cout << "  n_samples=" << n_samples << endl;
  //cout << "  n_dims=" << n_dims << endl;
  //cout << "  n_cost_vars=" << n_cost_vars << endl;
  //cout << "  cost_vars.cols()=" << cost_vars.cols() << endl;
  //cout << "  n_time_steps=" << n_time_steps << endl;

  costs1.resize(n_samples);
  costs2.resize(n_samples);

  // y_1..y_D  yd_1..yd_D  ydd_1..ydd_D  t=0
  // y_1..y_D  yd_1..yd_D  ydd_1..ydd_D  t=1
  //  |    |     |     |      |     |     |
  // y_1..y_D  yd_1..yd_D  ydd_1..ydd_D  t=T
  MatrixXd rollout; //(n_time_steps,n_cost_vars);
  MatrixXd my_row(1,n_time_steps*n_cost_vars);


 string dir1 = "/home/hrcstudents/Documents/Code/catkin_ws/ResultsUpdate/Goal1";
  string dir2 = "/home/hrcstudents/Documents/Code/catkin_ws/ResultsUpdate/Goal2";
 std::stringstream ss;
 ss << update;
 string str = ss.str();
 std::string text;
  MatrixXd costs_detailed1 = MatrixXd::Zero(n_samples,5); //human_time , robot_time, accuracy, acceleration, jerk
 MatrixXd costs_detailed2 = MatrixXd::Zero(n_samples,5); //human_time , robot_time, accuracy, acceleration, jerk
  //Costs variables
   VectorXd ts;
       double robot_time;
       int goal_time_step;
  MatrixXd y_after_goal ;
        double radius_goal = 0.01;
         double distance_to_goal;
             double human_time ;
              double accuracy;
 int a;
//ros::Time begin;
MatrixXd ydd;
MatrixXd yd;
    double sum_ydd = 0.0;
     double jerk_sum = 0.0;
         double time_step;
         MatrixXd y;
         MatrixXd forcing_terms;
geometry_msgs::Point msg;
//keyUp=n.subscribe("keyup", 10,getFeedback(human_time,accuracy,begin,keyUp,a));
//keyDown=n.subscribe("keydown", 10,getFeedback(human_time,accuracy,begin,keyDown,a));

//Subscribe

   ros::Rate loop_rate(1000);
         int i1=0;
         int i2=0;
 
         Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
  while ((i1+i2)<2*n_samples)
  {
          a = rand()%2 +1;
          sum_ydd = 0.0;
          jerk_sum = 0.0;

     if(a==1 && i1<n_samples){

         cout<<"here6" <<endl;
//costs_detailed = MatrixXd::Zero(n_samples,5);
    my_row = cost_vars1.row(i1);
    rollout = (Map<MatrixXd>(my_row.data(),n_cost_vars,n_time_steps)).transpose();

    cout << "rollout matrix: " << rollout.format(CleanFmt) << endl;
    //VectorXd DsByDx(n_time_steps-1);
  /*  for(double hh= 0; hh< ts.size()-1; hh++){
        double x1=rollout(hh,0);
        double x2=rollout(hh+1,0);
        double y1=rollout(hh,1);
        double y2=rollout(hh+1,1);
    //    DsByDx(hh)=sqrt(1+((y2-y1)/(x2-x1)).pow(2));
    }*/

//cout << rollout <<endl;
    // rollout is of size   n_time_steps x n_cost_vars
     ts = rollout.col(3 * n_dims);

 usleep(1000000);
 //begin= ros::Time::now();
  for(double hh= 0; hh< ts.size(); hh++) {
                  /**
      * This is a message object. You stuff it with data, and then publish it.
     */
      msg.x=rollout(hh,0)*0.1;
   msg.y=rollout(hh,1)*0.1;
    msg.z=0;
     //ROS_INFO("%d", msg.data);
   /**
+       * The publish() function is how you send messages. The parameter
+       * is the message object. The type of this object must agree with the type
+       * given as a template parameter to the advertise<>() call, as was done
+       * in the constructor above.
+       */

      traj_pub.publish(msg);
     ros::spinOnce();
     loop_rate.sleep();
        }     
     usleep(1000000);
      msg.x=0;
      msg.y=0;
     msg.z=0;
     // ROS_INFO("%d", msg.data);
    traj_pub.publish(msg);
     ros::spinOnce();
      loop_rate.sleep();
 
  robot_time = 0.0;
    if (robot_time_weight_!=0.0)
    {
      goal_time_step = 0;
      while (goal_time_step < ts.size() && ts[goal_time_step] < goal_time_)
        goal_time_step++;

      y_after_goal = rollout.block(goal_time_step, 0,rollout.rows() - goal_time_step, n_dims);


      //double time_of_goal= 0;

      for (int i=0; i< y_after_goal.rows(); i++){
          distance_to_goal = sqrt((y_after_goal.row(i) - goal_one_.transpose()).array().pow(2).sum());
          if (distance_to_goal <= radius_goal)
{robot_time= ts[goal_time_step+i+1];
              break;}
          else robot_time = goal_time_;
      } }
    costs_detailed1(i1,1)= robot_time_weight_*(robot_time-2)/1.5;
   // costs_detailed1(i1,1)= robot_time;
cout<< "robot time: " << robot_time <<endl;
   //plotting the rollout


        /*   Gnuplot gp2;
          cout << "plotting velocity of sample" << k << endl ;

       std::vector<std::pair<double, double> > xy_pts_C;
       for(double hh= 0; hh< ts.size(); hh++) {
          double x=rollout(hh,3 * n_dims);

           double y = rollout(hh,2);
          //  cout << "x is  " << x <<"  y is  "<< y <<endl;
           xy_pts_C.push_back(std::make_pair(x, y));
       }

              std::vector<std::pair<double, double> > xy_pts_D;

              for(double hh= 0; hh< ts.size(); hh++) {
                 double x=rollout(hh,3 * n_dims);

                  double y = rollout(hh,3);
                 //  cout << "x is  " << x <<"  y is  "<< y <<endl;
                  xy_pts_D.push_back(std::make_pair(x, y));
              }

              gp2 << "set autoscale \n";

              gp2 << "plot '-' with lines title 'dmp velocity1','-' with lines title 'dmp velocity2' \n" ;
              gp2.send1d(xy_pts_C);
                gp2.send1d(xy_pts_D); */



 /*Gnuplot gp3;

 cout << "plotting acceleration of sample" << k << endl ;

std::vector<std::pair<double, double> > xy_pts_E;
for(double hh= 0; hh< ts.size(); hh++) {
 double x=rollout(hh,3 * n_dims);

  double y = rollout(hh,4);
 //  cout << "x is  " << x <<"  y is  "<< y <<endl;
  xy_pts_E.push_back(std::make_pair(x, y));
}

     std::vector<std::pair<double, double> > xy_pts_F;

     for(double hh= 0; hh< ts.size(); hh++) {
        double x=rollout(hh,3 * n_dims);

         double y = rollout(hh,5);
        //  cout << "x is  " << x <<"  y is  "<< y <<endl;
         xy_pts_F.push_back(std::make_pair(x, y));
     }

     gp3 << "set autoscale \n";

     gp3 << "plot '-' with lines title 'dmp acceleration1' , '-' with lines title 'dmp acceleration2'\n" ;
     gp3.send1d(xy_pts_E);
       gp3.send1d(xy_pts_F); */

       cout << "plotting rollout of sample" << i1 << endl ;
       Gnuplot gp11;
           // For debugging or manual editing of commands:
           //Gnuplot gp(std::fopen("plot.gnu"));
           // or
           //Gnuplot gp("tee plot.gnu | gnuplot -persist");

           std::vector<std::pair<double, double> > xy_pts_A1;
           for(double hh= 0; hh< ts.size(); hh++) {
              double x=rollout(hh,0);

               double y = rollout(hh,1);
              //  cout << "x is  " << x <<"  y is  "<< y <<endl;
               xy_pts_A1.push_back(std::make_pair(x, y));
           }
     std::vector<std::pair<double, double> > xy_pts_B1;
     xy_pts_B1.push_back(std::make_pair(goal_one_[0], goal_one_[1]));
     xy_pts_B1.push_back(std::make_pair(goal_two_[0], goal_two_[1]));
           gp11 << "set xrange [-3:5]\nset yrange [-2:8]\n";

           gp11 << "plot '-' with lines title 'dmp', '-' with points title 'goals'\n";
     //   gp11<<   "set output 'home/ndonia/Documents/Codes/Freek/src/dmp_bbo/demos/demoDmpBbo/Goal1/output.eps'";
           gp11.send1d(xy_pts_A1);
             gp11.send1d(xy_pts_B1);


    cout << "enter human prediction time estimate for sample: " << i1 <<endl;
    cin >> human_time;
    cout << "human prediction time = " << human_time <<endl;
    costs_detailed1(i1,0)= human_time_weight_*(human_time-1)/time_weight;
//costs_detailed1(i1,0)= human_time;
    cout << "enter accuracy value 1 if false or 0 if true: ";
    cin >> accuracy;
    cout << "accuracy value = " << accuracy <<endl;
 costs_detailed1(i1,2)= accuracy_weight_*accuracy;




    ydd = rollout.block(0,2*n_dims,n_time_steps,n_dims);

    yd=rollout.block(0,n_dims,n_time_steps,n_dims);

    if (acceleration_weight_!=0.0)
    {
     // ydd = rollout.block(0,2*n_dims,n_time_steps,n_dims);
      // ydd = n_time_steps x n_dims
      sum_ydd = ydd.array().abs().sum();
       costs_detailed1(i1,3)= acceleration_weight_*(sum_ydd-3000)/1500;
    }


     time_step = ts[1]-ts[0];
    if (jerk_weight_!=0.0){
        jerk_sum=0;
    for (int i =0; i<ts.size()-1; i++){
        jerk_sum += abs(ydd(i+1,0)-ydd(i,0))/time_step;
         jerk_sum += abs(ydd(i+1,1)-ydd(i,1))/time_step;

      }
     costs_detailed1(i1,4)=  jerk_weight_*(jerk_sum-1*pow(10,4))/4000;
//costs_detailed1(i1,4)=  jerk_sum;
     }



   //   robot_time = (y_after_goal.rowwise() - goal_.transpose()).rowwise().squaredNorm().sum();

 y = rollout.block(0, 0, rollout.rows(), n_dims);
 forcing_terms= rollout.block(0,7,rollout.rows(), n_dims);


cout << "acceleration: " << ydd.format(CleanFmt) << endl;
cout << "velocity " << yd.format(CleanFmt) << endl;
cout << "position " << y.format(CleanFmt) << endl;
cout << "forcing terms " << forcing_terms.format(CleanFmt) << endl;
std::cout << costs_detailed1.format(CleanFmt) << endl;

  text="G1CostsDetailedUpdate"+str+".txt";
 saveMatrixHrc(dir1,text,costs_detailed1, true);

   /*   costs[k] =
     human_time_weight_* human_time +
      accuracy_weight_*accuracy +  acceleration_weight_*sum_ydd +
            robot_time_weight_*robot_time + jerk_weight_*jerk_sum;
                    */
costs1[i1] =
     human_time_weight_* (human_time-1)/time_weight+ accuracy_weight_*accuracy + acceleration_weight_* (sum_ydd-3000)/1500 +  robot_time_weight_*(robot_time-2)/1.5 +  jerk_weight_*(jerk_sum-1*pow(10,4))/4000;
//6*pow(10,4)

i1++;
  }


  if(a==2 && i2<n_samples){


      my_row = cost_vars2.row(i2);
      rollout = (Map<MatrixXd>(my_row.data(),n_cost_vars,n_time_steps)).transpose();

      Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
      cout << "rollout matrix: " << rollout.format(CleanFmt) << endl;
      //VectorXd DsByDx(n_time_steps-1);
    /*  for(double hh= 0; hh< ts.size()-1; hh++){
          double x1=rollout(hh,0);
          double x2=rollout(hh+1,0);
          double y1=rollout(hh,1);
          double y2=rollout(hh+1,1);
      //    DsByDx(hh)=sqrt(1+((y2-y1)/(x2-x1)).pow(2));
      }*/

  //cout << rollout <<endl;
      // rollout is of size   n_time_steps x n_cost_vars
       ts = rollout.col(3 * n_dims);

       usleep(1000000);
 //begin= ros::Time::now();
  for(double hh= 0; hh< ts.size(); hh++) {
                  /**
      * This is a message object. You stuff it with data, and then publish it.
     */
      msg.x=rollout(hh,0)*0.1;
   msg.y=rollout(hh,1)*0.1;
    msg.z=0;
    // ROS_INFO("%d", msg.data);
   /**
+       * The publish() function is how you send messages. The parameter
+       * is the message object. The type of this object must agree with the type
+       * given as a template parameter to the advertise<>() call, as was done
+       * in the constructor above.
+       */

      traj_pub.publish(msg);
     ros::spinOnce();
      loop_rate.sleep();
        }     
     usleep(1000000);
      msg.x=0;
      msg.y=0;
     msg.z=0;
      //ROS_INFO("%d", msg.data);
    traj_pub.publish(msg);
     ros::spinOnce();
      loop_rate.sleep();

      if (robot_time_weight_!=0.0)
      { goal_time_step = 0;

        while (goal_time_step < ts.size() && ts[goal_time_step] < goal_time_)
          goal_time_step++;

     y_after_goal = rollout.block(goal_time_step, 0,
          rollout.rows() - goal_time_step, n_dims);


        //double time_of_goal= 0;

        for (int i=0; i< y_after_goal.rows(); i++){
            distance_to_goal = sqrt((y_after_goal.row(i) - goal_two_.transpose()).array().pow(2).sum());
            if (distance_to_goal <= radius_goal)
  {robot_time= ts[goal_time_step+i+1];
                break;}
            else robot_time = goal_time_;
        } }
      costs_detailed2(i2,1)= robot_time_weight_*(robot_time-2)/1.5;
      //costs_detailed2(i2,1)= robot_time;
  cout<< "robot time: " << robot_time <<endl;
     //plotting the rollout


          /*   Gnuplot gp2;
            cout << "plotting velocity of sample" << k << endl ;

         std::vector<std::pair<double, double> > xy_pts_C;
         for(double hh= 0; hh< ts.size(); hh++) {
            double x=rollout(hh,3 * n_dims);

             double y = rollout(hh,2);
            //  cout << "x is  " << x <<"  y is  "<< y <<endl;
             xy_pts_C.push_back(std::make_pair(x, y));
         }

                std::vector<std::pair<double, double> > xy_pts_D;

                for(double hh= 0; hh< ts.size(); hh++) {
                   double x=rollout(hh,3 * n_dims);

                    double y = rollout(hh,3);
                   //  cout << "x is  " << x <<"  y is  "<< y <<endl;
                    xy_pts_D.push_back(std::make_pair(x, y));
                }

                gp2 << "set autoscale \n";

                gp2 << "plot '-' with lines title 'dmp velocity1','-' with lines title 'dmp velocity2' \n" ;
                gp2.send1d(xy_pts_C);
                  gp2.send1d(xy_pts_D); */



   /*Gnuplot gp3;

   cout << "plotting acceleration of sample" << k << endl ;

  std::vector<std::pair<double, double> > xy_pts_E;
  for(double hh= 0; hh< ts.size(); hh++) {
   double x=rollout(hh,3 * n_dims);

    double y = rollout(hh,4);
   //  cout << "x is  " << x <<"  y is  "<< y <<endl;
    xy_pts_E.push_back(std::make_pair(x, y));
  }

       std::vector<std::pair<double, double> > xy_pts_F;

       for(double hh= 0; hh< ts.size(); hh++) {
          double x=rollout(hh,3 * n_dims);

           double y = rollout(hh,5);
          //  cout << "x is  " << x <<"  y is  "<< y <<endl;
           xy_pts_F.push_back(std::make_pair(x, y));
       }

       gp3 << "set autoscale \n";

       gp3 << "plot '-' with lines title 'dmp acceleration1' , '-' with lines title 'dmp acceleration2'\n" ;
       gp3.send1d(xy_pts_E);
         gp3.send1d(xy_pts_F); */

         cout << "plotting rollout of sample" << i2 << endl ;
         Gnuplot gp22;
             // For debugging or manual editing of commands:
             //Gnuplot gp(std::fopen("plot.gnu"));
             // or
             //Gnuplot gp("tee plot.gnu | gnuplot -persist");

             std::vector<std::pair<double, double> > xy_pts_A2;
             for(double hh= 0; hh< ts.size(); hh++) {
                double x=rollout(hh,0);

                 double y = rollout(hh,1);
                //  cout << "x is  " << x <<"  y is  "<< y <<endl;
                 xy_pts_A2.push_back(std::make_pair(x, y));
             }
       std::vector<std::pair<double, double> > xy_pts_B2;
       xy_pts_B2.push_back(std::make_pair(goal_one_[0], goal_one_[1]));
       xy_pts_B2.push_back(std::make_pair(goal_two_[0], goal_two_[1]));
             gp22 << "set xrange [-3:5]\nset yrange [-2:8]\n";

             gp22 << "plot '-' with lines title 'dmp', '-' with points title 'goals'\n";
             gp22.send1d(xy_pts_A2);
               gp22.send1d(xy_pts_B2);


      cout << "enter human prediction time estimate for sample: " << i2 <<endl;
      cin >> human_time;
      cout << "human prediction time = " << human_time <<endl;
      costs_detailed2(i2,0)= human_time_weight_*(human_time-1)/time_weight;
//costs_detailed2(i2,0)= human_time;
      cout << "enter accuracy value 1 if false or 0 if true: ";
      cin >> accuracy;
      cout << "accuracy value = " << accuracy <<endl;
   costs_detailed2(i2,2)= accuracy_weight_*accuracy;




      ydd = rollout.block(0,2*n_dims,n_time_steps,n_dims);

      yd=rollout.block(0,n_dims,n_time_steps,n_dims);

      if (acceleration_weight_!=0.0)
      {
       // ydd = rollout.block(0,2*n_dims,n_time_steps,n_dims);
        // ydd = n_time_steps x n_dims
        sum_ydd = ydd.array().abs().sum();
         costs_detailed2(i2,3)= acceleration_weight_*(sum_ydd-3000)/1500;
      }


    time_step = ts[1]-ts[0];
      if (jerk_weight_!=0.0){
          jerk_sum=0;
      for (int i =0; i<ts.size()-1; i++){
          jerk_sum += abs(ydd(i+1,0)-ydd(i,0))/time_step;
           jerk_sum += abs(ydd(i+1,1)-ydd(i,1))/time_step;

        }
      costs_detailed2(i2,4)=  jerk_weight_*(jerk_sum-1*pow(10,4))/4000;
//costs_detailed2(i2,4)=  jerk_sum;
       }



     //   robot_time = (y_after_goal.rowwise() - goal_.transpose()).rowwise().squaredNorm().sum();

 y = rollout.block(0, 0, rollout.rows(), n_dims);
 forcing_terms= rollout.block(0,7,rollout.rows(), n_dims);


  cout << "acceleration: " << ydd.format(CleanFmt) << endl;
  cout << "velocity " << yd.format(CleanFmt) << endl;
  cout << "position " << y.format(CleanFmt) << endl;
  cout << "forcing terms " << forcing_terms.format(CleanFmt) << endl;
  std::cout << costs_detailed2.format(CleanFmt) << endl;


 text="G2CostsDetailedUpdate"+str+".txt";
   saveMatrixHrc(dir2,text,costs_detailed2, true);

     /*   costs[k] =
       human_time_weight_* human_time +
        accuracy_weight_*accuracy +  acceleration_weight_*sum_ydd +
              robot_time_weight_*robot_time + jerk_weight_*jerk_sum;
                      */
  costs2[i2] =
       human_time_weight_*(human_time-1)/time_weight+ accuracy_weight_*accuracy + acceleration_weight_* (sum_ydd-3000)/1500 +  robot_time_weight_*(robot_time-2)/1.5 +  jerk_weight_*(jerk_sum-1*pow(10,4))/4000;
  //6*pow(10,4)



i2++;
}
  }

  std::cout <<"costs 1 = " << costs1.format(CleanFmt) << endl;
    std::cout << "costs 2 = "<< costs2.format(CleanFmt) << endl;

}

void TaskViapoint::setCostFunctionWeighting(double viapoint_weight, double acceleration_weight, double goal_weight)
{
  viapoint_weight_      = viapoint_weight;
  acceleration_weight_  = acceleration_weight;
  goal_weight_          = goal_weight;
}

void TaskViapoint::setCostFunctionWeightingHrc(double human_time_weight, double robot_time_weight, double accuracy_weight, double acceleration_weight, double jerk_weight)
{
  human_time_weight_      = human_time_weight;
 robot_time_weight_  = robot_time_weight;
  accuracy_weight_          = accuracy_weight;
  acceleration_weight_ = acceleration_weight;
  jerk_weight_= jerk_weight;
}

void TaskViapoint::generateDemonstration(const MatrixXd& task_parameters, const VectorXd& ts, Trajectory& demonstration) const
{
  int n_dims = viapoint_.size();

  assert(task_parameters.rows()==1);
  assert(task_parameters.cols()==n_dims);

    VectorXd y_from    = VectorXd::Constant(n_dims,0.0);
    VectorXd y_to      = goal_;

    VectorXd y_yd_ydd_viapoint(3*n_dims);
    y_yd_ydd_viapoint << task_parameters.row(0), VectorXd::Constant(n_dims,1.0), VectorXd::Constant(n_dims,0.0);

  demonstration = Trajectory::generatePolynomialTrajectoryThroughViapoint(ts, y_from, y_yd_ydd_viapoint, viapoint_time_, y_to);

}


template<class Archive>
void TaskViapoint::serialize(Archive & ar, const unsigned int version)
{
  // serialize base class information
  ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(TaskWithTrajectoryDemonstrator);

  ar & BOOST_SERIALIZATION_NVP(viapoint_);
  ar & BOOST_SERIALIZATION_NVP(viapoint_time_);
  ar & BOOST_SERIALIZATION_NVP(viapoint_radius_);
  ar & BOOST_SERIALIZATION_NVP(goal_);
  ar & BOOST_SERIALIZATION_NVP(goal_time_);
  ar & BOOST_SERIALIZATION_NVP(viapoint_weight_);
  ar & BOOST_SERIALIZATION_NVP(acceleration_weight_);
  ar & BOOST_SERIALIZATION_NVP(goal_weight_);

  // new parameters
    ar & BOOST_SERIALIZATION_NVP(goal_one_);
      ar & BOOST_SERIALIZATION_NVP(goal_two_);
     ar & BOOST_SERIALIZATION_NVP(goal_selected_);
          ar & BOOST_SERIALIZATION_NVP(accuracy_weight_);
            ar & BOOST_SERIALIZATION_NVP(robot_time_weight_);
              ar & BOOST_SERIALIZATION_NVP(human_time_weight_);
                            ar & BOOST_SERIALIZATION_NVP(jerk_weight_);

}


string TaskViapoint::toString(void) const {
  RETURN_STRING_FROM_BOOST_SERIALIZATION_XML("TaskViapoint");
}


bool TaskViapoint::savePerformRolloutsPlotScript(string directory) const
{
  string filename = directory + "/plotRollouts.py";

  std::ofstream file;
  file.open(filename.c_str());
  if (!file.is_open())
  {
    std::cerr << "Couldn't open file '" << filename << "' for writing." << std::endl;
    return false;
  }

  file << "import numpy as np" << endl;
  file << "import matplotlib.pyplot as plt" << endl;
  file << "import sys, os" << endl;
  file << "def plotRollouts(cost_vars,ax):" << endl;
  file << "    viapoint = [";
  file << fixed;
  for (int ii=0; ii<viapoint_.size(); ii++)
  {
    if (ii>0) file << ", ";
    file << viapoint_[ii];
  }
  file << "]" << endl;
  file << "    viapoint_time = "<< viapoint_time_<< endl;
  file << "    # y      yd     ydd    1  forcing" << endl;
  file << "    # n_dofs n_dofs n_dofs ts n_dofs" << endl;
  file << "    n_dims = "<<viapoint_.size()<<";" << endl;
  file << "    n_vars = (1+n_dims*4)" << endl;
  file << "    if (len(cost_vars.shape)==1):" << endl;
  file << "        K=1;" << endl;
  file << "        n_cost_vars = len(cost_vars)" << endl;
  file << "        cost_vars = np.reshape(cost_vars,(K,n_cost_vars))" << endl;
  file << "    else:" << endl;
  file << "        K = len(cost_vars);" << endl;
  file << "        n_cost_vars = len(cost_vars[0])" << endl;
  file << "    n_time_steps = n_cost_vars/n_vars;" << endl;
  file << "    for k in range(len(cost_vars)):" << endl;
  file << "        x = np.reshape(cost_vars[k,:],(n_time_steps, n_vars))" << endl;
  file << "        y = x[:,0:n_dims]" << endl;
  file << "        t = x[:,3*n_dims]" << endl;
  if (viapoint_.size()==1)
  {
    file << "        line_handles = ax.plot(t,y,linewidth=0.5)" << endl;
    file << "    ax.plot(viapoint_time,viapoint,'ok')" << endl;
  }
  else
  {
    file << "        line_handles = ax.plot(y[:,0],y[:,1],linewidth=0.5)" << endl;
    file << "    ax.plot(viapoint[0],viapoint[1],'ok')" << endl;
  }
  file << "    return line_handles" << endl;
  file << "if __name__=='__main__':" << endl;
  file << "    # See if input directory was passed" << endl;
  file << "    if (len(sys.argv)==2):" << endl;
  file << "      directory = str(sys.argv[1])" << endl;
  file << "    else:" << endl;
  file << "      print 'Usage: '+sys.argv[0]+' <directory>';" << endl;
  file << "      sys.exit()" << endl;
  file << "    cost_vars = np.loadtxt(directory+\"cost_vars.txt\")" << endl;
  file << "    fig = plt.figure()" << endl;
  file << "    ax = fig.gca()" << endl;
  file << "    plotRollouts(cost_vars,ax)" << endl;
  file << "    plt.show()" << endl;

  file.close();

  return true;
}

bool TaskViapoint::savePerformRolloutsPlotScriptHrc(string directory) const
{
    string dir1 = "/home/ndonia/Documents/Codes/Freek/src/dmp_bbo/demos/demoDmpBbo/Goal1";
     string dir2 = "/home/ndonia/Documents/Codes/Freek/src/dmp_bbo/demos/demoDmpBbo/Goal2";
    string filename1 = dir1 + "/plotRollouts.py";
 string filename2 = dir2 + "/plotRollouts.py";

  std::ofstream file1;
  file1.open(filename1.c_str());
  if (!file1.is_open())
  {
    std::cerr << "Couldn't open file '" << filename1 << "' for writing." << std::endl;
    return false;
  }

  file1 << "import numpy as np" << endl;
  file1 << "import matplotlib.pyplot as plt" << endl;
  file1 << "import sys, os" << endl;
  file1 << "def plotRollouts(cost_vars,ax):" << endl;

  file1 << "    goal_one = [";
  file1 << fixed;
  for (int ii=0; ii<goal_one_.size(); ii++)
  {
    if (ii>0) file1 << ", ";
    file1 << goal_one_[ii];
  }
  file1 << "]" << endl;

  file1 << "    goal_two = [";
  file1 << fixed;
  for (int ii=0; ii<goal_two_.size(); ii++)
  {
    if (ii>0) file1 << ", ";
    file1 << goal_two_[ii];
  }
  file1 << "]" << endl;


  file1 << "    goal_time =  "<< goal_time_ << endl;
  file1 << "    # y      yd     ydd    1  forcing" << endl;
  file1 << "    # n_dofs n_dofs n_dofs ts n_dofs" << endl;
  file1 << "    n_dims = "<<viapoint_.size()<<";" << endl;
  file1 << "    n_vars = (1+n_dims*4)" << endl;
  file1 << "    if (len(cost_vars.shape)==1):" << endl;
  file1 << "        K=1;" << endl;
  file1 << "        n_cost_vars = len(cost_vars)" << endl;
  file1 << "        cost_vars = np.reshape(cost_vars,(K,n_cost_vars))" << endl;
  file1 << "    else:" << endl;
  file1 << "        K = len(cost_vars);" << endl;
  file1 << "        n_cost_vars = len(cost_vars[0])" << endl;
  file1 << "    n_time_steps = n_cost_vars/n_vars;" << endl;
  file1 << "    for k in range(len(cost_vars)):" << endl;
  file1 << "        x = np.reshape(cost_vars[k,:],(n_time_steps, n_vars))" << endl;
  file1 << "        y = x[:,0:n_dims]" << endl;
  file1 << "        t = x[:,3*n_dims]" << endl;

  if (goal_one_.size()==1)
  {
    file1 << "        line_handles = ax.plot(t,y,linewidth=0.5)" << endl;
    file1 << "    ax.plot(goal_time,goal_one,'ok')" << endl;
    file1 << "    ax.plot(goal_time,goal_two,'ok')" << endl;
    //goal time is 0.9
  }
  else
  {
    file1 << "        line_handles = ax.plot(y[:,0],y[:,1],linewidth=0.5)" << endl;
    file1 << "    ax.plot(goal_one[0],goal_one[1],'ok')" << endl;
     file1 << "    ax.plot(goal_two[0],goal_two[1],'ok')" << endl;
  }
  file1 << "    return line_handles" << endl;


  file1 << "if __name__=='__main__':" << endl;
  file1 << "    # See if input directory was passed" << endl;
  file1 << "    if (len(sys.argv)==2):" << endl;
  file1 << "      directory = str(sys.argv[1])" << endl;
  file1 << "    else:" << endl;
  file1 << "      print 'Usage: '+sys.argv[0]+' <directory>';" << endl;
  file1 << "      sys.exit()" << endl;
  file1 << "    cost_vars = np.loadtxt(directory+\"cost_vars.txt\")" << endl;
  file1 << "    fig = plt.figure()" << endl;
  file1 << "    ax = fig.gca()" << endl;
  file1 << "    plotRollouts(cost_vars,ax)" << endl;
  file1 << "    plt.show()" << endl;

  file1.close();


  //Goal2
  std::ofstream file2;
  file2.open(filename2.c_str());
  if (!file2.is_open())
  {
    std::cerr << "Couldn't open file '" << filename2 << "' for writing." << std::endl;
    return false;
  }

  file2 << "import numpy as np" << endl;
  file2 << "import matplotlib.pyplot as plt" << endl;
  file2 << "import sys, os" << endl;
  file2 << "def plotRollouts(cost_vars,ax):" << endl;

  file2 << "    goal_one = [";
  file2 << fixed;
  for (int ii=0; ii<goal_one_.size(); ii++)
  {
    if (ii>0) file2 << ", ";
    file2 << goal_one_[ii];
  }
  file2<< "]" << endl;

  file2 << "    goal_two = [";
  file2 << fixed;
  for (int ii=0; ii<goal_two_.size(); ii++)
  {
    if (ii>0) file2 << ", ";
    file2 << goal_two_[ii];
  }
  file2 << "]" << endl;


  file2 << "    goal_time =  "<< goal_time_ << endl;
  file2 << "    # y      yd     ydd    1  forcing" << endl;
  file2 << "    # n_dofs n_dofs n_dofs ts n_dofs" << endl;
  file2 << "    n_dims = "<<viapoint_.size()<<";" << endl;
  file2 << "    n_vars = (1+n_dims*4)" << endl;
  file2 << "    if (len(cost_vars.shape)==1):" << endl;
  file2 << "        K=1;" << endl;
  file2 << "        n_cost_vars = len(cost_vars)" << endl;
  file2 << "        cost_vars = np.reshape(cost_vars,(K,n_cost_vars))" << endl;
  file2 << "    else:" << endl;
  file2 << "        K = len(cost_vars);" << endl;
  file2 << "        n_cost_vars = len(cost_vars[0])" << endl;
  file2 << "    n_time_steps = n_cost_vars/n_vars;" << endl;
  file2<< "    for k in range(len(cost_vars)):" << endl;
  file2 << "        x = np.reshape(cost_vars[k,:],(n_time_steps, n_vars))" << endl;
  file2 << "        y = x[:,0:n_dims]" << endl;
  file2 << "        t = x[:,3*n_dims]" << endl;

  if (goal_one_.size()==1)
  {
    file2 << "        line_handles = ax.plot(t,y,linewidth=0.5)" << endl;
    file2 << "    ax.plot(goal_time,goal_one,'ok')" << endl;
    file2 << "    ax.plot(goal_time,goal_two,'ok')" << endl;
    //goal time is 0.9
  }
  else
  {
    file2 << "        line_handles = ax.plot(y[:,0],y[:,1],linewidth=0.5)" << endl;
    file2 << "    ax.plot(goal_one[0],goal_one[1],'ok')" << endl;
     file2 << "    ax.plot(goal_two[0],goal_two[1],'ok')" << endl;
  }
  file2 << "    return line_handles" << endl;


  file2 << "if __name__=='__main__':" << endl;
  file2 << "    # See if input directory was passed" << endl;
  file2 << "    if (len(sys.argv)==2):" << endl;
  file2 << "      directory = str(sys.argv[1])" << endl;
  file2 << "    else:" << endl;
  file2 << "      print 'Usage: '+sys.argv[0]+' <directory>';" << endl;
  file2 << "      sys.exit()" << endl;
  file2 << "    cost_vars = np.loadtxt(directory+\"cost_vars.txt\")" << endl;
  file2 << "    fig = plt.figure()" << endl;
  file2 << "    ax = fig.gca()" << endl;
  file2 << "    plotRollouts(cost_vars,ax)" << endl;
  file2 << "    plt.show()" << endl;

  file2.close();
  return true;
}

/*bool TaskViapoint::savePerformRolloutsPlotScriptHrc(string directory) const
{
    string dir1 = "/home/ndonia/Documents/Codes/Freek/src/dmp_bbo/demos/demoDmpBbo/Goal1";
     string dir2 = "/home/ndonia/Documents/Codes/Freek/src/dmp_bbo/demos/demoDmpBbo/Goal2";
    string filename1 = dir1 + "/plotRollouts.py";
 string filename2 = dir2 + "/plotRollouts.py";

  std::ofstream file1;
  file1.open(filename1.c_str());
  if (!file1.is_open())
  {
    std::cerr << "Couldn't open file '" << filename1 << "' for writing." << std::endl;
    return false;
  }

  file1 << "import numpy as np" << endl;
  file1 << "import matplotlib.pyplot as plt" << endl;
  file1<< "import sys, os" << endl;
  file1 << "def plotRollouts(cost_vars1,cost_vars2,ax):" << endl;

  file1 << "    goal_one = [";
  file1 << fixed;
  for (int ii=0; ii<goal_one_.size(); ii++)
  {
    if (ii>0) file << ", ";
    file1 << goal_one_[ii];
  }
  file1 << "]" << endl;

  file1 << "    goal_two = [";
  file1 << fixed;
  for (int ii=0; ii<goal_two_.size(); ii++)
  {
    if (ii>0) file << ", ";
    file1 << goal_two_[ii];
  }
  file1 << "]" << endl;



  //file << "    viapoint_time = "<< viapoint_time_<< endl;

  file1 << "    goal_time =  "<< goal_time_ << endl;

  file1 << "    # y      yd     ydd    1  forcing" << endl;
  file1<< "    # n_dofs n_dofs n_dofs ts n_dofs" << endl;
  file1 << "    n_dims = "<<goal_one_.size()<<";" << endl;
  file1<< "    n_vars = (1+n_dims*4)" << endl;
  file1 << "    if (len(cost_vars1.shape)==1):" << endl;
  file1 << "        K1=1;" << endl;
  file1 << "        n_cost_vars1 = len(cost_vars1)" << endl;
  file1 << "        cost_vars1 = np.reshape(cost_vars1,(K1,n_cost_vars1))" << endl;
  file1 << "    else:" << endl;
  file1 << "        K1 = len(cost_vars1);" << endl;
  file1 << "        n_cost_vars1 = len(cost_vars1[0])" << endl;
  file1 << "    n_time_steps1 = n_cost_vars1/n_vars;" << endl;
    file1 << "    n_time_steps2 = n_cost_vars2/n_vars;" << endl;
  file1 << "    for k in range(len(cost_vars1)):" << endl;
  file1 << "        x = np.reshape(cost_vars1[k,:],(n_time_steps1, n_vars))" << endl;
  file1 << "        y = x[:,0:n_dims]" << endl;
  file1 << "        t = x[:,3*n_dims]" << endl;



  if (goal_one_.size()==1)
  {
    file1 << "        line_handles = ax.plot(t,y,linewidth=0.5)" << endl;
    file1 << "    ax.plot(goal_time,goal_one,'ok')" << endl;
    file1 << "    ax.plot(goal_time,goal_two,'ok')" << endl;
    //goal time is 0.9
  }
  else
  {
    file1 << "        line_handles = ax.plot(y[:,0],y[:,1],linewidth=0.5)" << endl;
    file1 << "    ax.plot(goal_one[0],goal_one[1],'ok')" << endl;
     file1 << "    ax.plot(goal_two[0],goal_two[1],'ok')" << endl;
  }
  file1 << "    return line_handles" << endl;

  //new plot





  if (goal_one_.size()==1)
  {
    file << "        line_handles = ax.plot(t,y,linewidth=0.5)" << endl;

  }
  else
  {
    file << "        line_handles = ax.plot(y[:,0],y[:,1],linewidth=0.5)" << endl;

  }
  file << "    return line_handles" << endl;



  file << "if __name__=='__main__':" << endl;
  file << "    # See if input directory was passed" << endl;
  file << "    if (len(sys.argv)==2):" << endl;
  file << "      directory = str(sys.argv[1])" << endl;
  file << "    else:" << endl;
  file << "      print 'Usage: '+sys.argv[0]+' <directory>';" << endl;
  file << "      sys.exit()" << endl;
  file << "    cost_vars1 = np.loadtxt(directory+\"cost_vars1.txt\")" << endl;
    file << "    cost_vars2 = np.loadtxt(directory+\"cost_vars2.txt\")" << endl;
  file << "    fig = plt.figure()" << endl;
  file << "    ax = fig.gca()" << endl;
  file << "    plotRollouts(cost_vars1,cost_vars1,ax)" << endl;






  file << "    plt.show()" << endl;

  file.close();

  return true;
}*/


}
