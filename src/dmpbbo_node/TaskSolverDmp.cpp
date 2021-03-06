/**
 * @file   TaskSolverDmp.cpp
 * @brief  TaskSolverDmp class source file.
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

#include "/home/hrcstudents/Documents/dmpbbo_ws/src/predictable_kuka_v1/include/dmp_bbo_node/TaskSolverDmp.hpp"





#include <boost/serialization/export.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include "/home/hrcstudents/Documents/Code/catkin_ws/src/dmp_bbo_node/include/dmp_bbo_node/TaskSolverDmp.hpp"

/** For boost::serialization. See http://www.boost.org/doc/libs/1_55_0/libs/serialization/doc/special.html#export */
BOOST_CLASS_EXPORT_IMPLEMENT(DmpBbo::TaskSolverDmp);

#include <boost/serialization/base_object.hpp>

#include <iostream>
#include <string>
#include <set>
#include <eigen3/Eigen/Core>

#include "/home/hrcstudents/Documents/Code/catkin_ws/src/dmp_bbo_node/include/dmp_bbo_node/EigenBoostSerialization.hpp"
#include "/home/hrcstudents/Documents/Code/catkin_ws/src/dmp_bbo_node/include/dmp_bbo_node/BoostSerializationToString.hpp"
#include "/home/hrcstudents/Documents/dmpbbo_ws/src/predictable_kuka_v1/include/dmp_bbo_node/Dmp.hpp"


#include "/home/hrcstudents/Documents/Code/catkin_ws/src/dmp_bbo_node/include/dmp_bbo_node/EigenFileIO.hpp"

#include <vector>
#include <map>
#include <limits>
#include <cmath>
#include <cstdio>
//#include <boost/tuple/tuple.hpp>
//#include <boost/foreach.hpp>
#include "/home/hrcstudents/Documents/Code/catkin_ws/src/dmp_bbo_node/include/dmp_bbo_node/gnuplot-iostream.h"

using namespace std;
using namespace Eigen;
int s = 1 ;
string saveDir = "/home/hrcstudents/Documents/Code/catkin_ws/ResultsUpdate/Goal1";
std::string testaya ;


namespace DmpBbo {
  
TaskSolverDmp::TaskSolverDmp(Dmp* dmp, std::set<std::string> optimize_parameters, double dt, double integrate_dmp_beyond_tau_factor, bool use_normalized_parameter)
: dmp_(dmp)
{
  dmp_->setSelectedParameters(optimize_parameters);
  
 integrate_time_ = dmp_->tau() * integrate_dmp_beyond_tau_factor;
 //ntegrate_time_ = dmp_->tau() ;
  n_time_steps_ = (integrate_time_/dt)+1;
  use_normalized_parameter_ = use_normalized_parameter;
}

void TaskSolverDmp::set_perturbation(double perturbation_standard_deviation)
{
  dmp_->set_perturbation_analytical_solution(perturbation_standard_deviation);
}


void TaskSolverDmp::performRollouts(const Eigen::MatrixXd& samples, Eigen::MatrixXd& cost_vars,  double avoidNum) const
{

  // samples         = n_samples x sum(nmodel_parameters_)
  // task_parameters = n_samples x n_task_pars
  // cost_vars       = n_samples x (n_time_steps*n_cost_vars)
  
  int n_dofs = dmp_->dim_orig();
  int n_samples = samples.rows();

  VectorXd ts = VectorXd::LinSpaced(n_time_steps_,0.0,integrate_time_);
  
  int n_cost_vars = 4*n_dofs+1;
  cost_vars.resize(n_samples,n_time_steps_*n_cost_vars); 
    
  VectorXd model_parameters;
  for (int k=0; k<n_samples; k++)
  {
    model_parameters = samples.row(k);
    dmp_->setParameterVectorSelected(model_parameters, use_normalized_parameter_);
    
    int n_dims = dmp_->dim(); // Dimensionality of the system
    MatrixXd xs_ana(n_time_steps_,n_dims);
    MatrixXd xds_ana(n_time_steps_,n_dims);
    MatrixXd forcing_terms(n_time_steps_,n_dofs);
    forcing_terms.fill(0.0);
    dmp_->analyticalSolution(ts,xs_ana,xds_ana,forcing_terms,avoidNum);
    /*    
    // Here's the non-analytical version, which is slower.
    MatrixXd xs(n_time_steps_,n_dims);
    MatrixXd xds(n_time_steps_,n_dims);
    
    // Use integrateStart to get the initial x and xd
    VectorXd x(n_dims);
    VectorXd xd(n_dims);
    dmp_->integrateStart(x,xd);
    xs.row(0)  = x.transpose();
    xds.row(0)  = xd.transpose();
    // Use DynamicalSystemSystem::integrateStep to integrate numerically step-by-step
    double dt = ts[1]-ts[0];
    for (int ii=1; ii<n_time_steps_; ii++)
    {
      dmp_->integrateStep(dt,x,           x,          xd); 
      //                     previous x   updated x   updated xd
      xs.row(ii)  = x.transpose();
      xds.row(ii)  = xd.transpose();
    }
    MatrixXd ys;
    MatrixXd yds;
    MatrixXd ydds;
    dmp_->statesAsTrajectory(xs,xds,ys,yds,ydds);
    */

    
    MatrixXd ys_ana;
    MatrixXd yds_ana;
    MatrixXd ydds_ana;
    dmp_->statesAsTrajectory(xs_ana,xds_ana,ys_ana,yds_ana,ydds_ana);


   /* //plotting the rollout
     cout << "plotting rollout of sample" << k << endl ;
     Gnuplot gp;
         // For debugging or manual editing of commands:
         //Gnuplot gp(std::fopen("plot.gnu"));
         // or
         //Gnuplot gp("tee plot.gnu | gnuplot -persist");

         std::vector<std::pair<double, double> > xy_pts_A;
         for(double hh= 0; hh< ts.size(); hh++) {
            double x=ys_ana(hh,0);
             double y = ys_ana(hh,1);
             xy_pts_A.push_back(std::make_pair(x, y));
         }
  // std::vector<std::pair<double, double> > xy_pts_B;
  // xy_pts_B.push_back(std::make_pair(goal_one_[0], goal_one_[1]));
   //xy_pts_B.push_back(std::make_pair(goal_two_[0], goal_two_[1]));
         gp << "set autoscale\n";
         gp << "plot '-' with lines title 'dmp', '-' with points title 'goals'\n";
         gp.send1d(xy_pts_A);
          // gp.send1d(xy_pts_B);
*/  

    int offset = 0;
    for (int tt=0; tt<n_time_steps_; tt++)
    {

      cost_vars.block(k,offset,1,n_dofs) = ys_ana.row(tt);   offset += n_dofs; 

    //  cout << ys_ana.row(tt) <<endl;
      
      cost_vars.block(k,offset,1,n_dofs) = yds_ana.row(tt);  offset += n_dofs; 
      cost_vars.block(k,offset,1,n_dofs) = ydds_ana.row(tt); offset += n_dofs; 
      cost_vars.block(k,offset,1,1) = ts.row(tt);       offset += 1; 
      cost_vars.block(k,offset,1,n_dofs) = forcing_terms.row(tt); offset += n_dofs;       
    }
  }
}
















void TaskSolverDmp::performRollouts(const Eigen::MatrixXd& samples, Eigen::MatrixXd& cost_vars) const
{

  // samples         = n_samples x sum(nmodel_parameters_)
  // task_parameters = n_samples x n_task_pars
  // cost_vars       = n_samples x (n_time_steps*n_cost_vars)
  
  int n_dofs = dmp_->dim_orig();
  int n_samples = samples.rows();

  VectorXd ts = VectorXd::LinSpaced(n_time_steps_,0.0,integrate_time_);
  
  int n_cost_vars = 4*n_dofs+1;
  cost_vars.resize(n_samples,n_time_steps_*n_cost_vars); 
    
  VectorXd model_parameters;
  for (int k=0; k<n_samples; k++)
  {
    model_parameters = samples.row(k);

    dmp_->setParameterVectorSelected(model_parameters, use_normalized_parameter_);

    int n_dims = dmp_->dim(); // Dimensionality of the system
    MatrixXd xs_ana(n_time_steps_,n_dims);
    MatrixXd xds_ana(n_time_steps_,n_dims);
    MatrixXd forcing_terms(n_time_steps_,n_dofs);
    forcing_terms.fill(0.0);

    dmp_->analyticalSolution(ts,xs_ana,xds_ana,forcing_terms);

    /*    
    // Here's the non-analytical version, which is slower.
    MatrixXd xs(n_time_steps_,n_dims);
    MatrixXd xds(n_time_steps_,n_dims);
    
    // Use integrateStart to get the initial x and xd
    VectorXd x(n_dims);
    VectorXd xd(n_dims);
    dmp_->integrateStart(x,xd);
    xs.row(0)  = x.transpose();
    xds.row(0)  = xd.transpose();
    // Use DynamicalSystemSystem::integrateStep to integrate numerically step-by-step
    double dt = ts[1]-ts[0];
    for (int ii=1; ii<n_time_steps_; ii++)
    {
      dmp_->integrateStep(dt,x,           x,          xd); 
      //                     previous x   updated x   updated xd
      xs.row(ii)  = x.transpose();
      xds.row(ii)  = xd.transpose();
    }
    MatrixXd ys;
    MatrixXd yds;
    MatrixXd ydds;
    dmp_->statesAsTrajectory(xs,xds,ys,yds,ydds);
    */

    
    MatrixXd ys_ana;
    MatrixXd yds_ana;
    MatrixXd ydds_ana;
    dmp_->statesAsTrajectory(xs_ana,xds_ana,ys_ana,yds_ana,ydds_ana);

    

   /* //plotting the rollout
     cout << "plotting rollout of sample" << k << endl ;
     Gnuplot gp;
         // For debugging or manual editing of commands:
         //Gnuplot gp(std::fopen("plot.gnu"));
         // or
         //Gnuplot gp("tee plot.gnu | gnuplot -persist");

         std::vector<std::pair<double, double> > xy_pts_A;
         for(double hh= 0; hh< ts.size(); hh++) {
            double x=ys_ana(hh,0);
             double y = ys_ana(hh,1);
             xy_pts_A.push_back(std::make_pair(x, y));
         }
  // std::vector<std::pair<double, double> > xy_pts_B;
  // xy_pts_B.push_back(std::make_pair(goal_one_[0], goal_one_[1]));
   //xy_pts_B.push_back(std::make_pair(goal_two_[0], goal_two_[1]));
         gp << "set autoscale\n";
         gp << "plot '-' with lines title 'dmp', '-' with points title 'goals'\n";
         gp.send1d(xy_pts_A);
          // gp.send1d(xy_pts_B);
*/

    int offset = 0;
    for (int tt=0; tt<n_time_steps_; tt++)
    {
      
      cost_vars.block(k,offset,1,n_dofs) = ys_ana.row(tt);   offset += n_dofs; 
     
    //  cout << ys_ana.row(tt) <<endl;
      cost_vars.block(k,offset,1,n_dofs) = yds_ana.row(tt);  offset += n_dofs; 
      cost_vars.block(k,offset,1,n_dofs) = ydds_ana.row(tt); offset += n_dofs; 
      cost_vars.block(k,offset,1,1) = ts.row(tt);       offset += 1; 
      cost_vars.block(k,offset,1,n_dofs) = forcing_terms.row(tt); offset += n_dofs;       
    }
  }
}

void TaskSolverDmp::performRollouts(const vector<MatrixXd>& samples, MatrixXd& cost_vars) const 
{

  // n_dofs-D Dmp, n_parallel=n_dofs
  //                   vector<Matrix>            Matrix
  // samples         = n_dofs x          n_samples x sum(nmodel_parameters_)
  // task_parameters =                   n_samples x n_task_pars
  // cost_vars       =                   n_samples x (n_time_steps*n_cost_vars)
  
  int n_dofs = samples.size();
  assert(n_dofs>0);
  assert(n_dofs==dmp_->dim_orig());
  
  int n_samples = samples[0].rows();
  for (int dd=1; dd<n_dofs; dd++)
  {
    assert(samples[dd].rows()==n_samples);
  }

  VectorXd ts = VectorXd::LinSpaced(n_time_steps_,0.0,integrate_time_);
  
  int n_cost_vars = 4*n_dofs+1;
  cost_vars.resize(n_samples,n_time_steps_*n_cost_vars); 
    
  vector<VectorXd> model_parameters_vec(n_dofs);
  for (int k=0; k<n_samples; k++)
  {
    for (int dd=0; dd<n_dofs; dd++)
      model_parameters_vec[dd] = samples[dd].row(k);
    dmp_->setParameterVectorSelected(model_parameters_vec, use_normalized_parameter_);
    
    int n_dims = dmp_->dim(); // Dimensionality of the system
    MatrixXd xs_ana(n_time_steps_,n_dims);
    MatrixXd xds_ana(n_time_steps_,n_dims);
    MatrixXd forcing_terms(n_time_steps_,n_dofs);
    forcing_terms.fill(0.0);
    dmp_->analyticalSolution(ts,xs_ana,xds_ana,forcing_terms);

    /*    
    // Here's the non-analytical version, which is slower.
    MatrixXd xs(n_time_steps_,n_dims);
    MatrixXd xds(n_time_steps_,n_dims);
    
    // Use integrateStart to get the initial x and xd
    VectorXd x(n_dims);
    VectorXd xd(n_dims);
    dmp_->integrateStart(x,xd);
    xs.row(0)  = x.transpose();
    xds.row(0)  = xd.transpose();
    // Use DynamicalSystemSystem::integrateStep to integrate numerically step-by-step
    double dt = ts[1]-ts[0];
    for (int ii=1; ii<n_time_steps_; ii++)
    {
      dmp_->integrateStep(dt,x,           x,          xd); 
      //                     previous x   updated x   updated xd
      xs.row(ii)  = x.transpose();
      xds.row(ii)  = xd.transpose();
    }
    MatrixXd ys;
    MatrixXd yds;
    MatrixXd ydds;
    dmp_->statesAsTrajectory(xs,xds,ys,yds,ydds);
    */

    
    MatrixXd ys_ana;
    MatrixXd yds_ana;
    MatrixXd ydds_ana;
    dmp_->statesAsTrajectory(xs_ana,xds_ana,ys_ana,yds_ana,ydds_ana);
    
    int offset = 0;
    for (int tt=0; tt<n_time_steps_; tt++)
    {
      cost_vars.block(k,offset,1,n_dofs) = ys_ana.row(tt);   offset += n_dofs; 
      cost_vars.block(k,offset,1,n_dofs) = yds_ana.row(tt);  offset += n_dofs; 
      cost_vars.block(k,offset,1,n_dofs) = ydds_ana.row(tt); offset += n_dofs; 
      cost_vars.block(k,offset,1,1) = ts.row(tt);       offset += 1; 
      cost_vars.block(k,offset,1,n_dofs) = forcing_terms.row(tt); offset += n_dofs;       
    }
  }
}




















string TaskSolverDmp::toString(void) const
{
  RETURN_STRING_FROM_BOOST_SERIALIZATION_XML("TaskSolverDmp");
}

}
