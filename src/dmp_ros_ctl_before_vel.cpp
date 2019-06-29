/**
 *
 * DmpBbo is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with DmpBbo.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <string>
#include <set>
#include <eigen3/Eigen/Core>

#include "/home/hrcstudents/Documents/Code/catkin_ws/src/dmp_bbo_node/include/dmp_bbo_node/TaskSolver.hpp"
#include "/home/hrcstudents/Documents/Code/catkin_ws/src/dmp_bbo_node/include/dmp_bbo_node/TaskSolverDmp.hpp"
#include "/home/hrcstudents/Documents/Code/catkin_ws/src/dmp_bbo_node/include/dmp_bbo_node/Task.hpp"
#include "/home/hrcstudents/Documents/Code/catkin_ws/src/dmp_bbo_node/include/dmp_bbo_node/Trajectory.hpp"
#include "/home/hrcstudents/Documents/Code/catkin_ws/src/dmp_bbo_node/include/dmp_bbo_node/Dmp.hpp"
#include "/home/hrcstudents/Documents/Code/catkin_ws/src/dmp_bbo_node/include/dmp_bbo_node/ModelParametersLWR.hpp"
#include "/home/hrcstudents/Documents/Code/catkin_ws/src/dmp_bbo_node/include/dmp_bbo_node/FunctionApproximatorLWR.hpp"
#include "/home/hrcstudents/Documents/Code/catkin_ws/src/dmp_bbo_node/include/dmp_bbo_node/ModelParametersRBFN.hpp"
#include "/home/hrcstudents/Documents/Code/catkin_ws/src/dmp_bbo_node/include/dmp_bbo_node/MetaParametersRBFN.hpp"
#include "/home/hrcstudents/Documents/Code/catkin_ws/src/dmp_bbo_node/include/dmp_bbo_node/FunctionApproximatorRBFN.hpp"
#include "/home/hrcstudents/Documents/Code/catkin_ws/src/dmp_bbo_node/include/dmp_bbo_node/DistributionGaussian.hpp"
#include "/home/hrcstudents/Documents/Code/catkin_ws/src/dmp_bbo_node/include/dmp_bbo_node/Updater.hpp"
#include "/home/hrcstudents/Documents/Code/catkin_ws/src/dmp_bbo_node/include/dmp_bbo_node/UpdaterCovarDecay.hpp"
#include "/home/hrcstudents/Documents/Code/catkin_ws/src/dmp_bbo_node/include/dmp_bbo_node/SpringDamperSystem.hpp"
#include "/home/hrcstudents/Documents/Code/catkin_ws/src/dmp_bbo_node/include/dmp_bbo_node/ExponentialSystem.hpp"
#include "/home/hrcstudents/Documents/Code/catkin_ws/src/dmp_bbo_node/include/dmp_bbo_node/TimeSystem.hpp"
#include "/home/hrcstudents/Documents/Code/catkin_ws/src/dmp_bbo_node/include/dmp_bbo_node/SigmoidSystem.hpp"
#include "/home/hrcstudents/Documents/Code/catkin_ws/src/dmp_bbo_node/include/dmp_bbo_node/EigenFileIO.hpp"
#include "/home/hrcstudents/Documents/Code/catkin_ws/src/dmp_bbo_node/include/dmp_bbo_node/UpdateSummary.hpp"
#include "ros/ros.h"
#include "std_msgs/String.h"
#include "std_msgs/Float32.h"
#include <sstream>
#include "geometry_msgs/Point.h"
#include <vector>
#include <map>
#include <limits>
#include <cmath>
#include <cstdio>
#include "/home/hrcstudents/Documents/Code/catkin_ws/src/dmp_bbo_node/include/dmp_bbo_node/gnuplot-iostream.h"
#include <thread>
#include <chrono>
#include "/home/hrcstudents/Documents/Code/catkin_ws/src/dmp_bbo_node/include/dmp_bbo_node/keyboard.h"
#include "/usr/include/SDL/SDL.h"
#include <keyboard/Key.h>
#include <eigen3/Eigen/Geometry>
#include "sensor_msgs/JointState.h"
#include "trajectory_msgs/JointTrajectory.h"
#include "trajectory_msgs/JointTrajectoryPoint.h"
using namespace std;
using namespace Eigen;
using namespace DmpBbo;

double human_time;
double human_time_default = 3.0;
double accuracy;
int a;
double beginTime;
double robot_time;
double end;
VectorXd goal1(7);
VectorXd goal2(7);
VectorXd goal3(7);
VectorXd y_init(7);
VectorXd ts;
int goal_time_step;
MatrixXd ydd;
MatrixXd yd;
double sum_ydd = 0.0;
double jerk_sum = 0.0;
double jerk_sum_2 = 0.0;
double time_step;
MatrixXd y;
MatrixXd forcing_terms;
double time_weight;
int counterMessage = 0;
int counterTraj = 0;
double accuracy_weight_;
double robot_time_weight_;
double human_time_weight_;
double jerk_weight_;
double jerk_weight_2_;
double obstacle_avoidance_weight_;
double trust_region_weight_;
double trust = 0;
double goal_time_;
const double PI = 3.141592653589793238463;
double q1, q2, q3, q4, q5, q6, q7;
int sampleSize = 4;     // was 1: sampling each discretized point on the dmp rollout
int pointer = 0;
int test = 1;
bool flag = false;
int returnNum = -200;
MatrixXd ptompData(51, 3);
bool ptompSet = false;
ros::Publisher traj_pub;
ros::Publisher traj_pub2;
sensor_msgs::JointState joint_state;
std::chrono::milliseconds delayTraj(1500);
std::chrono::milliseconds publishing(20);
std::chrono::milliseconds avoiding(50);
std::chrono::seconds breakTime(5);
string predictedGoal;
int universalCount1 = 1;
int universalCount2 = 1;
int universalCount3 = 1;
int obstacle_cost = 0;

string dir1 = "/home/hrcstudents/Documents/Code/catkin_ws/ResultsUpdate/Goal1";
string dir2 = "/home/hrcstudents/Documents/Code/catkin_ws/ResultsUpdate/Goal2";
string dir3 = "/home/hrcstudents/Documents/Code/catkin_ws/ResultsUpdate/Goal3";

void computeCosts(Eigen::MatrixXd& costs_detailed, Eigen::MatrixXd& rollout, Eigen::MatrixXd& rolloutFK, Eigen::VectorXd& costs, int a, int ii, double i_update, Eigen::VectorXd& ts, string dir, VectorXd& goal, int n_samples);
void getPtomp(const sensor_msgs::JointState::ConstPtr& msg);
void getTime(const std_msgs::Float32::ConstPtr& msg);
Eigen::MatrixXd forwardKinematic(double q1, double q2, double q3, double q4, double q5, double q6, double q7);
int checkCollision(Eigen::MatrixXd ts, Eigen::MatrixXd rolloutFKAvoidance);
int executeAvoidance(MatrixXd rollout, MatrixXd rolloutFK, int hh, int returnNum, trajectory_msgs::JointTrajectory);

int main(int n_args, char* args[])
{

    //Arguments for simulation on VREP

    std::string trajTopic; //KUKA Topic

    trajTopic = "/standing2/joint_trajectory_controller/command";
    ros::Time::useSystemTime();
    //initialize ROS subscriber
    std::string nodeName("dmp_bbo");
    ros::init(n_args, args, nodeName.c_str());
    ros::NodeHandle node;
    ros::Subscriber getPtompSub; //Get the PTOMP Data Online
    ros::Subscriber getTimeSub; //Get the Human Prdiction Time and TRUST
    printf("dmp_bbo has just started please fasten your seat belts %s\n", nodeName.c_str());
    getPtompSub = node.subscribe<sensor_msgs::JointState>("/JS_predict_wrist", 10, getPtomp);
    getTimeSub = node.subscribe<std_msgs::Float32>("/motion_duration", 10, getTime);
    traj_pub = node.advertise<trajectory_msgs::JointTrajectory>(trajTopic.c_str(), 4000); //Kuka Publisher
    ros::Rate loop_rate(25);   //was 100: change publish rate for testing


    //Must initialize the joint_state parameters, Velocity is set as a constant for all joints expept J1
    //Change in case ROS Control works
    trajectory_msgs::JointTrajectory msg_;

    msg_.joint_names.clear();
    msg_.joint_names.push_back("standing2_a1_joint");
    msg_.joint_names.push_back("standing2_a2_joint");
    msg_.joint_names.push_back("standing2_a3_joint");
    msg_.joint_names.push_back("standing2_a4_joint");
    msg_.joint_names.push_back("standing2_a5_joint");
    msg_.joint_names.push_back("standing2_a6_joint");
    msg_.joint_names.push_back("standing2_e1_joint");
    msg_.points.resize(1);
    msg_.points[0].positions.resize(7); 
    msg_.points[0].time_from_start = ros::Duration(0.015*sampleSize);
   

    //DMP Parameters and Task Initiation

    int n_dims = 7;

    //DMP Trained in Radians

    goal1 << 40 * PI / 180, 70 * PI / 180, 50 * PI / 180, -95 * PI / 180, 0 * PI / 180, 0, 0;
    goal2 << 35 * PI / 180.0, 80 * PI / 180.0, 60 * PI / 180.0, -70 * PI / 180.0, 0 * PI / 180.0, 0 * PI / 180.0, 0;
    goal3 << 60 * PI / 180.0, 90 * PI / 180.0, 70 * PI / 180.0, -75 * PI / 180.0, 0 * PI / 180.0, 0 * PI / 180.0, 0;
    y_init << 20 * PI / 180.0, 50 * PI / 180.0, 50 * PI / 180.0, -60 * PI / 180.0, 0 * PI / 180.0, 0, 0;

    // Some DMP parameters
    human_time = 3.0;
    accuracy = 0;
    double tau = 2.5;
    double integrate_dmp_beyond_tau_factor = 1.2;
    goal_time_ = tau;
    double dt = 0.01;
    VectorXd y_attr = goal1;
    accuracy_weight_ = 10;
    human_time_weight_ = 10;
    trust_region_weight_ = 2;
    jerk_weight_ = 1;
    jerk_weight_2_ = 1;
    obstacle_avoidance_weight_ = 2;
    int n_updates = 5;
    int n_samples_per_update = 5;
    double covarSize = 100; //Important as search space determines variations
    bool plotOrNo = false;
    bool pressEnter = false;
    bool takeBreak = true;

    cout << "Press enter to start the experiment" << endl;
    cin.ignore();
    std::this_thread::sleep_for(std::chrono::seconds(1));

    // Make the initial function approximators
    int n_basis_functions = 21;
    double intersection_height = 0.9;
    bool overwrite = true;
    double alpha_spring_damper = 20;
    int n_time_steps = (tau / dt) + 1;
    VectorXd ts = VectorXd::LinSpaced(n_time_steps, 0.0, tau); // Time steps

    //MetaParameters only work when training the trajectory, a more suitable FA are model parameters as you get to assign the centers and weights directly but they dont give you the flexibility to train specific trajectories

    MetaParametersRBFN* meta_parameters1 = new MetaParametersRBFN(1, n_basis_functions, intersection_height);
    MetaParametersRBFN* meta_parameters2 = new MetaParametersRBFN(1, n_basis_functions, intersection_height);
    MetaParametersRBFN* meta_parameters3 = new MetaParametersRBFN(1, n_basis_functions, intersection_height);

    FunctionApproximatorRBFN* fa_rbfn1 = new FunctionApproximatorRBFN(meta_parameters1);
    FunctionApproximatorRBFN* fa_rbfn2 = new FunctionApproximatorRBFN(meta_parameters2);
    FunctionApproximatorRBFN* fa_rbfn3 = new FunctionApproximatorRBFN(meta_parameters3);

    vector<FunctionApproximator*> function_approximators1(n_dims);
    vector<FunctionApproximator*> function_approximators2(n_dims);
    vector<FunctionApproximator*> function_approximators3(n_dims);

    for (int i_dim = 0; i_dim < n_dims; i_dim++)
        function_approximators1[i_dim] = fa_rbfn1->clone();
    for (int i_dim = 0; i_dim < n_dims; i_dim++)
        function_approximators2[i_dim] = fa_rbfn2->clone();
    for (int i_dim = 0; i_dim < n_dims; i_dim++)
        function_approximators3[i_dim] = fa_rbfn3->clone();

    VectorXd one_1 = VectorXd::Ones(1);
    VectorXd one_0 = VectorXd::Zero(1);
    DynamicalSystem* goal_system1 = new ExponentialSystem(tau, y_init, goal1, 15);
    DynamicalSystem* goal_system2 = new ExponentialSystem(tau, y_init, goal2, 15);
    DynamicalSystem* goal_system3 = new ExponentialSystem(tau, y_init, goal3, 15);

    DynamicalSystem* phase_system = new TimeSystem(tau, false);
    DynamicalSystem* gating_system = new ExponentialSystem(tau, one_1, one_0, 5);

    //Train the DMPs with a Min Jerk Trajectory

    Trajectory trajectory1;
    Trajectory trajectory2;
    Trajectory trajectory3;

    trajectory1 = Trajectory::generateMinJerkTrajectory(ts, y_init, goal1);
    trajectory2 = Trajectory::generateMinJerkTrajectory(ts, y_init, goal2);
    trajectory3 = Trajectory::generateMinJerkTrajectory(ts, y_init, goal3);

    Dmp* dmp1 = new Dmp(n_dims, function_approximators1, alpha_spring_damper, goal_system1, phase_system, gating_system);
    Dmp* dmp2 = new Dmp(n_dims, function_approximators2, alpha_spring_damper, goal_system2, phase_system, gating_system);
    Dmp* dmp3 = new Dmp(n_dims, function_approximators3, alpha_spring_damper, goal_system3, phase_system, gating_system);

    dmp1->train(trajectory1, dir1, false);
    dmp2->train(trajectory2, dir2, false);
    dmp3->train(trajectory3, dir3, false);

    set<string> parameters_to_optimize;
    parameters_to_optimize.insert("weights");

    bool use_normalized_parameter = false;
    TaskSolverDmp* task_solver1 = new TaskSolverDmp(dmp1, parameters_to_optimize,
        dt, integrate_dmp_beyond_tau_factor, use_normalized_parameter);
    TaskSolverDmp* task_solver2 = new TaskSolverDmp(dmp2, parameters_to_optimize,
        dt, integrate_dmp_beyond_tau_factor, use_normalized_parameter);
    TaskSolverDmp* task_solver3 = new TaskSolverDmp(dmp3, parameters_to_optimize,
        dt, integrate_dmp_beyond_tau_factor, use_normalized_parameter);

    //Analytical Solution of our system containing all the information we need
    MatrixXd xs_ana, xds_ana, forcing_terms_ana, fa_output_ana;
    MatrixXd xs_ana2, xds_ana2, forcing_terms_ana2, fa_output_ana2;
    MatrixXd xs_ana3, xds_ana3, forcing_terms_ana3, fa_output_ana3;

    dmp1->analyticalSolution(ts, xs_ana, xds_ana, forcing_terms_ana, fa_output_ana);
    dmp2->analyticalSolution(ts, xs_ana2, xds_ana2, forcing_terms_ana2, fa_output_ana2);
    dmp3->analyticalSolution(ts, xs_ana3, xds_ana3, forcing_terms_ana3, fa_output_ana3);

    MatrixXd output_ana(ts.size(), 1 + xs_ana.cols() + xds_ana.cols());
    MatrixXd output_ana2(ts.size(), 1 + xs_ana2.cols() + xds_ana2.cols());
    MatrixXd output_ana3(ts.size(), 1 + xs_ana3.cols() + xds_ana3.cols());

    output_ana << xs_ana, xds_ana, ts;
    output_ana2 << xs_ana2, xds_ana2, ts;
    output_ana3 << xs_ana3, xds_ana3, ts;

    saveMatrix(dir1, "reproduced_xs_xds1.txt", output_ana, overwrite);
    saveMatrix(dir1, "reproduced_forcing_terms1.txt", forcing_terms_ana, overwrite);
    saveMatrix(dir1, "reproduced_fa_output1.txt", fa_output_ana, overwrite);

    saveMatrix(dir2, "reproduced_xs_xds2.txt", output_ana2, overwrite);
    saveMatrix(dir2, "reproduced_forcing_terms2.txt", forcing_terms_ana2, overwrite);
    saveMatrix(dir2, "reproduced_fa_output2.txt", fa_output_ana2, overwrite);

    // Make the initial distribution
    VectorXd mean_init1;
    VectorXd mean_init2;
    VectorXd mean_init3;

    dmp1->getParameterVectorSelected(mean_init1);
    dmp2->getParameterVectorSelected(mean_init2);
    dmp3->getParameterVectorSelected(mean_init3);

    MatrixXd covar_init1 = covarSize * MatrixXd::Identity(mean_init1.size(), mean_init1.size());
    MatrixXd covar_init2 = covarSize * MatrixXd::Identity(mean_init2.size(), mean_init2.size());
    MatrixXd covar_init3 = covarSize * MatrixXd::Identity(mean_init3.size(), mean_init3.size());

    DistributionGaussian* distribution1 = new DistributionGaussian(mean_init1, covar_init1);
    DistributionGaussian* distribution2 = new DistributionGaussian(mean_init2, covar_init2);
    DistributionGaussian* distribution3 = new DistributionGaussian(mean_init3, covar_init3);

    // Make the parameter updater
    double eliteness = 10;
    double covar_decay_factor = 0.6;
    string weighting_method("PI-BB");
    Updater* updater1 = new UpdaterCovarDecay(eliteness, covar_decay_factor, weighting_method);
    Updater* updater2 = new UpdaterCovarDecay(eliteness, covar_decay_factor, weighting_method);
    Updater* updater3 = new UpdaterCovarDecay(eliteness, covar_decay_factor, weighting_method);

    // Run the optimization

    string save_directory = "/home/hrcstudents/Documents/Code/catkin_ws/ResultsUpdate";
    Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
    MatrixXd samples1,samples2,samples3;
    MatrixXd cost_vars1, cost_vars2, cost_vars3, cost_vars4;
    MatrixXd cost_vars_eval1, cost_vars_eval2, cost_vars_eval3, cost_vars_eval4;
    VectorXd costs1, costs2, costs3, costs4;
    VectorXd cost_eval1, cost_eval2, cost_eval3, cost_eval4;
    // Bookkeeping
    vector<UpdateSummary> update_summaries1;
    vector<UpdateSummary> update_summaries2;
    vector<UpdateSummary> update_summaries3;
    vector<UpdateSummary> update_summaries4;

    vector<UpdateSummary> update_summaries1init;
    vector<UpdateSummary> update_summaries2init;
    vector<UpdateSummary> update_summaries3init;
    vector<UpdateSummary> update_summaries4init;

    UpdateSummary update_summary1;
    UpdateSummary update_summary2;
    UpdateSummary update_summary3;
    UpdateSummary update_summary4;

    bool only_learning_curve = false;

    std::string text;
    std::string s;

    //************
    //****************
    //************
    //****************
    //************
    //****************

    // Optimization loop

    for (int i_update = 0; i_update < n_updates; i_update++) {

        // 0. Get cost of current distribution mean
        task_solver1->performRollouts(distribution1->mean().transpose(), cost_vars_eval1);
        task_solver2->performRollouts(distribution2->mean().transpose(), cost_vars_eval2);
        task_solver3->performRollouts(distribution3->mean().transpose(), cost_vars_eval3);

        cout << "Update no. =  " << i_update << endl;

        for (int i = 0; i < 1; i++) {

            int n_samples = cost_vars_eval1.rows();
            int n_dims = goal1.size();
            int n_cost_vars = 4 * n_dims + 1; // y_1..y_D  yd_1..yd_D  ydd_1..ydd_D  t forcing_term_1..forcing_term_D
            int n_time_steps = cost_vars_eval1.cols() / n_cost_vars;

            time_weight = 2.5;

            cost_eval1.resize(n_samples);
            cost_eval2.resize(n_samples);
            cost_eval3.resize(n_samples);

            MatrixXd rollout; //(n_time_steps,n_cost_vars);
            MatrixXd angularRollout(n_time_steps, 5);
            MatrixXd rolloutFK(n_time_steps, 3);
            MatrixXd rolloutFKd(n_time_steps - 1, 3);
            MatrixXd rolloutFKAvoidance(n_time_steps, 3);

            MatrixXd my_row(1, n_time_steps * n_cost_vars);

            MatrixXd costs_detailed1 = MatrixXd::Zero(n_samples, 12); //human_time , robot_time, accuracy,  jerk, obstacle avoidance
            MatrixXd costs_detailed2 = MatrixXd::Zero(n_samples, 12); //human_time , robot_time, accuracy,  jerk, obstacle avoidance
            MatrixXd costs_detailed3 = MatrixXd::Zero(n_samples, 12); //human_time , robot_time, accuracy,  jerk, obstacle avoidance

            int i1 = 0;
            int i2 = 0;
            int i3 = 0;

            while ((i1 + i2 + i3) < 3 * n_samples) {

                a = rand() % 3 + 1;

                sum_ydd = 0.0;
                jerk_sum = 0.0;

                if (a == 1 && i1 < n_samples) {

                    cout << "Press enter to resume" << endl;
                    cin.ignore();
                    my_row = cost_vars_eval1.row(i1);
                    rollout = (Map<MatrixXd>(my_row.data(), n_cost_vars, n_time_steps)).transpose();
                    yd = rollout.block(0, n_dims, n_time_steps, n_dims);

                    ts = rollout.col(3 * n_dims);
                    time_step = ts[1] - ts[0];

                    //Creating FK matrix For comparison with PROMP

                    for (int i = 0; i < ts.size(); i++) {
                        q1 = rollout(i, 0);
                        q2 = rollout(i, 1);
                        q3 = rollout(i, 2);
                        q4 = rollout(i, 3);
                        q5 = rollout(i, 4);
                        q6 = rollout(i, 5);

                        rolloutFKAvoidance(i, 0) = forwardKinematic(q1, q2, q3, q4, q5, q6, q7)(0, 3);
                        rolloutFKAvoidance(i, 1) = forwardKinematic(q1, q2, q3, q4, q5, q6, q7)(1, 3);
                        rolloutFKAvoidance(i, 2) = forwardKinematic(q1, q2, q3, q4, q5, q6, q7)(2, 3);
                    }

                    cout << "Press enter to start trajectory G1" << endl;
                    beginTime = ros::Time::now().toSec();

                    for (int hh = 0; hh < ts.size(); hh += sampleSize) {

                        msg_.points[0].positions[0] = rollout(hh, 0) ;
                        msg_.points[0].positions[1] = rollout(hh, 1) ;
                        msg_.points[0].positions[6] = rollout(hh, 2) ;
                        msg_.points[0].positions[2] = rollout(hh, 3) ;
                        msg_.points[0].positions[3] = rollout(hh, 4) ;
                        msg_.points[0].positions[4] = rollout(hh, 5) ;
                        msg_.points[0].positions[5] = rollout(hh, 6) ;

                        q1 = rollout(hh, 0);
                        q2 = rollout(hh, 1);
                        q3 = rollout(hh, 2);
                        q4 = rollout(hh, 3);
                        q5 = rollout(hh, 4);
                        q6 = rollout(hh, 5);

                        rolloutFK(hh, 0) = forwardKinematic(q1, q2, q3, q4, q5, q6, q7)(0, 3);
                        rolloutFK(hh, 1) = forwardKinematic(q1, q2, q3, q4, q5, q6, q7)(1, 3);
                        rolloutFK(hh, 2) = forwardKinematic(q1, q2, q3, q4, q5, q6, q7)(2, 3);

                        traj_pub.publish(msg_);
                        ros::spinOnce();
                        loop_rate.sleep();

                        if (ptompSet) {
                            returnNum = checkCollision(ts, rolloutFKAvoidance);

                            if (returnNum > 250)
                                returnNum = 250;
                            ptompSet = false;
                        }

                        if (hh == returnNum - 30) {
                            flag = false;   // was true, but let's check without collision avoidance
                        }
                        if (flag == true) {
                            hh = executeAvoidance(rollout, rolloutFK, hh, returnNum, msg_);
                            flag = false;
                            returnNum = -200;
                        }
                    }

                    std::string text;
                    std::stringstream ss;
                    ss << universalCount1;
                    string str = ss.str();
                    universalCount1++;

                    for (int hh = ts.size(); hh > 0; hh -= sampleSize) {

                        msg_.points[0].positions[0] = rollout(hh - 1, 0) ;
                        msg_.points[0].positions[1] = rollout(hh - 1, 1) ;
                        msg_.points[0].positions[6] = rollout(hh - 1, 2) ;
                        msg_.points[0].positions[2] = rollout(hh - 1, 3) ;
                        msg_.points[0].positions[3] = rollout(hh - 1, 4) ;
                        msg_.points[0].positions[4] = rollout(hh - 1, 5) ;
                        msg_.points[0].positions[5] = rollout(hh - 1, 6) ;

                        traj_pub.publish(msg_);
                        ros::spinOnce();
                        loop_rate.sleep();
                    }

                    if (predictedGoal == "1")
                        accuracy = 0;
                    else
                        accuracy = 1;

                    cout << "humans time is:" << human_time << endl;
                    cout << "accuracy is:" << accuracy << endl;
                    computeCosts(costs_detailed1, rollout, rolloutFK, cost_eval1, a, i1, (i_update + 1) * 100, ts, dir1, goal1, n_samples);
                    i1++;
                    pointer++;
                }

                if (a == 2 && i2 < n_samples) {

                    cout << "Press enter to resume" << endl;
                    cin.ignore();
                    my_row = cost_vars_eval2.row(i2);
                    rollout = (Map<MatrixXd>(my_row.data(), n_cost_vars, n_time_steps)).transpose();
                    // rollout is of size   n_time_steps x n_cost_vars
                    ts = rollout.col(3 * n_dims);

                    for (int i = 0; i < ts.size(); i++) {
                        q1 = rollout(i, 0);
                        q2 = rollout(i, 1);
                        q3 = rollout(i, 2);
                        q4 = rollout(i, 3);
                        q5 = rollout(i, 4);
                        q6 = rollout(i, 5);

                        rolloutFKAvoidance(i, 0) = forwardKinematic(q1, q2, q3, q4, q5, q6, q7)(0, 3);
                        rolloutFKAvoidance(i, 1) = forwardKinematic(q1, q2, q3, q4, q5, q6, q7)(1, 3);
                        rolloutFKAvoidance(i, 2) = forwardKinematic(q1, q2, q3, q4, q5, q6, q7)(2, 3);
                    }

                    cout << "Press enter to start trajectory G2" << endl;
                    beginTime = ros::Time::now().toSec();

                    for (int hh = 0; hh < ts.size(); hh += sampleSize) {
                        msg_.points[0].positions[0] = rollout(hh, 0) ;
                        msg_.points[0].positions[1] = rollout(hh, 1) ;
                        msg_.points[0].positions[6] = rollout(hh, 2) ;
                        msg_.points[0].positions[2] = rollout(hh, 3) ;
                        msg_.points[0].positions[3] = rollout(hh, 4) ;
                        msg_.points[0].positions[4] = rollout(hh, 5) ;
                        msg_.points[0].positions[5] = rollout(hh, 6) ;

                        q1 = rollout(hh, 0);
                        q2 = rollout(hh, 1);
                        q3 = rollout(hh, 2);
                        q4 = rollout(hh, 3);
                        q5 = rollout(hh, 4);
                        q6 = rollout(hh, 5);

                        rolloutFK(hh, 0) = forwardKinematic(q1, q2, q3, q4, q5, q6, q7)(0, 3);
                        rolloutFK(hh, 1) = forwardKinematic(q1, q2, q3, q4, q5, q6, q7)(1, 3);
                        rolloutFK(hh, 2) = forwardKinematic(q1, q2, q3, q4, q5, q6, q7)(2, 3);

                        traj_pub.publish(msg_);
                        ros::spinOnce();
                        loop_rate.sleep();

                        if (ptompSet) {
                            returnNum = checkCollision(ts, rolloutFKAvoidance);

                            if (returnNum > 250)
                                returnNum = 250;
                            ptompSet = false;
                        }

                        if (hh == returnNum - 30) {
                            flag = false;
                        }
                        if (flag == true) {
                            hh = executeAvoidance(rollout, rolloutFK, hh, returnNum, msg_);
                            flag = false;
                            returnNum = -200;
                        }
                    }

                    std::string text;
                    std::stringstream ss;
                    ss << universalCount2;
                    string str = ss.str();
                    universalCount2++;

                    for (int hh = ts.size(); hh > 0; hh -= sampleSize) {

                        msg_.points[0].positions[0] = rollout(hh - 1, 0) ;
                        msg_.points[0].positions[1] = rollout(hh - 1, 1) ;
                        msg_.points[0].positions[6] = rollout(hh - 1, 2) ;
                        msg_.points[0].positions[2] = rollout(hh - 1, 3) ;
                        msg_.points[0].positions[3] = rollout(hh - 1, 4) ;
                        msg_.points[0].positions[4] = rollout(hh - 1, 5) ;
                        msg_.points[0].positions[5] = rollout(hh - 1, 6) ;

                        traj_pub.publish(msg_);
                        ros::spinOnce();
                        //loop_rate.sleep();
                        loop_rate.sleep();
                    }

                    if (predictedGoal == "0")
                        accuracy = 0;
                    else
                        accuracy = 1;
                    cout << "humans time is:" << human_time << endl;
                    cout << "accuracy is:" << accuracy << endl;
                    computeCosts(costs_detailed2, rollout, rolloutFK, cost_eval2, a, i2, (i_update + 1) * 100, ts, dir2, goal2, n_samples);

                    i2++;
                }

                if (a == 3 && i3 < n_samples) {

                    cout << "Press enter to resume" << endl;
                    cin.ignore();
                    my_row = cost_vars_eval3.row(i3);
                    rollout = (Map<MatrixXd>(my_row.data(), n_cost_vars, n_time_steps)).transpose();

                    // rollout is of size   n_time_steps x n_cost_vars
                    ts = rollout.col(3 * n_dims);

                    for (int i = 0; i < ts.size(); i++) {
                        q1 = rollout(i, 0);
                        q2 = rollout(i, 1);
                        q3 = rollout(i, 2);
                        q4 = rollout(i, 3);
                        q5 = rollout(i, 4);
                        q6 = rollout(i, 5);

                        rolloutFKAvoidance(i, 0) = forwardKinematic(q1, q2, q3, q4, q5, q6, q7)(0, 3);
                        rolloutFKAvoidance(i, 1) = forwardKinematic(q1, q2, q3, q4, q5, q6, q7)(1, 3);
                        rolloutFKAvoidance(i, 2) = forwardKinematic(q1, q2, q3, q4, q5, q6, q7)(2, 3);
                    }

                    cout << "Press enter to start trajectory G3" << endl;
                    beginTime = ros::Time::now().toSec();

                    for (int hh = 0; hh < ts.size(); hh += sampleSize) {
                        msg_.points[0].positions[0] = rollout(hh, 0) ;
                        msg_.points[0].positions[1] = rollout(hh, 1) ;
                        msg_.points[0].positions[6] = rollout(hh, 2) ;
                        msg_.points[0].positions[2] = rollout(hh, 3) ;
                        msg_.points[0].positions[3] = rollout(hh, 4) ;
                        msg_.points[0].positions[4] = rollout(hh, 5) ;
                        msg_.points[0].positions[5] = rollout(hh, 6) ;

                        traj_pub.publish(msg_);
                        ros::spinOnce();
                        loop_rate.sleep();

                        if (ptompSet) {
                            returnNum = checkCollision(ts, rolloutFKAvoidance);

                            if (returnNum > 250)
                                returnNum = 250;
                            ptompSet = false;
                        }

                        if (hh == returnNum - 30) {
                            flag = false;
                        }
                        if (flag == true) {
                            hh = executeAvoidance(rollout, rolloutFK, hh, returnNum, msg_);
                            flag = false;
                            returnNum = -200;
                        }
                    }

                    std::string text;
                    std::stringstream ss;
                    ss << universalCount3;
                    string str = ss.str();
                    universalCount3++;

                    for (int hh = ts.size(); hh > 0; hh -= sampleSize) {

                        msg_.points[0].positions[0] = rollout(hh - 1, 0) ;
                        msg_.points[0].positions[1] = rollout(hh - 1, 1) ;
                        msg_.points[0].positions[6] = rollout(hh - 1, 2) ;
                        msg_.points[0].positions[2] = rollout(hh - 1, 3) ;
                        msg_.points[0].positions[3] = rollout(hh - 1, 4) ;
                        msg_.points[0].positions[4] = rollout(hh - 1, 5) ;
                        msg_.points[0].positions[5] = rollout(hh - 1, 6) ;

                        traj_pub.publish(msg_);
                        ros::spinOnce();
                        loop_rate.sleep();
                    }

                    if (predictedGoal == "2")
                        accuracy = 0;
                    else
                        accuracy = 1;
                    cout << "humans time is:" << human_time << endl;
                    cout << "accuracy is:" << accuracy << endl;
                    computeCosts(costs_detailed3, rollout, rolloutFK, cost_eval3, a, i3, (i_update + 1) * 100, ts, dir3, goal3, n_samples);

                    i3++;
                }

                //FIRST PART IS DONE
                //****************************************
                //**************************** //****************************************
                //**************************** //****************************************
                //**************************** //****************************************
                //**************************** //****************************************
                //**************************** //****************************************
                //**************************** //****************************************
                //**************************** //****************************************
                //****************************
            }

            std::cout << "costs 1 = " << cost_eval1.format(CleanFmt) << endl;
            std::cout << "costs 2 = " << cost_eval2.format(CleanFmt) << endl;
            std::cout << "costs 3 = " << cost_eval3.format(CleanFmt) << endl;
        }

        //****************************************
        //****************************
        // 1. Sample from distribution

        distribution1->generateSamples(n_samples_per_update, samples1);
        distribution2->generateSamples(n_samples_per_update, samples2);
        distribution3->generateSamples(n_samples_per_update, samples3);

        // 2A. Perform the roll-outs
        task_solver1->performRollouts(samples1, cost_vars1);
        task_solver2->performRollouts(samples2, cost_vars2);
        task_solver3->performRollouts(samples3, cost_vars3);

        for (int i = 0; i < 1; i++) {
            int n_samples = cost_vars1.rows();
            int n_dims = goal1.size();
            int n_cost_vars = 4 * n_dims + 1; // y_1..y_D  yd_1..yd_D  ydd_1..ydd_D  t forcing_term_1..forcing_term_D
            int n_time_steps = cost_vars1.cols() / n_cost_vars;
            time_weight = 2.5;

            costs1.resize(n_samples);
            costs2.resize(n_samples);
            costs3.resize(n_samples);

            MatrixXd rollout;
            MatrixXd rolloutFK(n_time_steps, 3);
            MatrixXd rolloutFKAvoidance(n_time_steps, 3);
            MatrixXd my_row(1, n_time_steps * n_cost_vars);

            MatrixXd costs_detailed1 = MatrixXd::Zero(n_samples, 12); //human_time , robot_time, accuracy, jerk, obstacle avoidance
            MatrixXd costs_detailed2 = MatrixXd::Zero(n_samples, 12); //human_time , robot_time, accuracy, jerk, obstacle avoidance
            MatrixXd costs_detailed3 = MatrixXd::Zero(n_samples, 12); //human_time , robot_time, accuracy, jerk, obstacle avoidance

            int i1 = 0;
            int i2 = 0;
            int i3 = 0;

            while ((i1 + i2 + i3) < 3 * n_samples) {

                a = rand() % 3 + 1;
                sum_ydd = 0.0;
                jerk_sum = 0.0;

                if (a == 1 && i1 < n_samples) {

                    cout << "Press enter to resume" << endl;
                    cin.ignore();

                    my_row = cost_vars1.row(i1);
                    rollout = (Map<MatrixXd>(my_row.data(), n_cost_vars, n_time_steps)).transpose();

                    // rollout is of size   n_time_steps x n_cost_vars
                    ts = rollout.col(3 * n_dims);
                    for (int i = 0; i < ts.size(); i++) {
                        q1 = rollout(i, 0);
                        q2 = rollout(i, 1);
                        q3 = rollout(i, 2);
                        q4 = rollout(i, 3);
                        q5 = rollout(i, 4);
                        q6 = rollout(i, 5);

                        rolloutFKAvoidance(i, 0) = forwardKinematic(q1, q2, q3, q4, q5, q6, q7)(0, 3);
                        rolloutFKAvoidance(i, 1) = forwardKinematic(q1, q2, q3, q4, q5, q6, q7)(1, 3);
                        rolloutFKAvoidance(i, 2) = forwardKinematic(q1, q2, q3, q4, q5, q6, q7)(2, 3);
                    }

                    cout << "Press enter to start trajectory G1 " << endl;
                    beginTime = ros::Time::now().toSec();

                    for (int hh = 0; hh < ts.size(); hh += sampleSize) {
                        msg_.points[0].positions[0] = rollout(hh, 0) ;
                        msg_.points[0].positions[1] = rollout(hh, 1) ;
                        msg_.points[0].positions[6] = rollout(hh, 2) ;
                        msg_.points[0].positions[2] = rollout(hh, 3) ;
                        msg_.points[0].positions[3] = rollout(hh, 4) ;
                        msg_.points[0].positions[4] = rollout(hh, 5) ;
                        msg_.points[0].positions[5] = rollout(hh, 6) ;

                        q1 = rollout(hh, 0);
                        q2 = rollout(hh, 1);
                        q3 = rollout(hh, 2);
                        q4 = rollout(hh, 3);
                        q5 = rollout(hh, 4);
                        q6 = rollout(hh, 5);

                        rolloutFK(hh, 0) = forwardKinematic(q1, q2, q3, q4, q5, q6, q7)(0, 3);
                        rolloutFK(hh, 1) = forwardKinematic(q1, q2, q3, q4, q5, q6, q7)(1, 3);
                        rolloutFK(hh, 2) = forwardKinematic(q1, q2, q3, q4, q5, q6, q7)(2, 3);

                        traj_pub.publish(msg_);
                        ros::spinOnce();
                        loop_rate.sleep();

                        if (ptompSet) {
                            returnNum = checkCollision(ts, rolloutFKAvoidance);

                            if (returnNum > 250)
                                returnNum = 250;
                            ptompSet = false;
                        }

                        if (hh == returnNum - 30) {
                            flag = false;
                        }
                        if (flag == true) {
                            hh = executeAvoidance(rollout, rolloutFK, hh, returnNum, msg_);
                            flag = false;
                            returnNum = -200;
                        }
                    }

                    std::string text;
                    std::stringstream ss;
                    ss << universalCount1;
                    string str = ss.str();
                    universalCount1++;

                    for (int hh = ts.size(); hh > 0; hh -= sampleSize) {
                        msg_.points[0].positions[0] = rollout(hh - 1, 0) ;
                        msg_.points[0].positions[1] = rollout(hh - 1, 1) ;
                        msg_.points[0].positions[6] = rollout(hh - 1, 2) ;
                        msg_.points[0].positions[2] = rollout(hh - 1, 3) ;
                        msg_.points[0].positions[3] = rollout(hh - 1, 4) ;
                        msg_.points[0].positions[4] = rollout(hh - 1, 5) ;
                        msg_.points[0].positions[5] = rollout(hh - 1, 6) ;

                        traj_pub.publish(msg_);
                        ros::spinOnce();
                        loop_rate.sleep();
                    }

                    if (predictedGoal == "1")
                        accuracy = 0;
                    else
                        accuracy = 1;
                    cout << "humans time is:" << human_time << endl;
                    cout << "accuracy is:" << accuracy << endl;

                    computeCosts(costs_detailed1, rollout, rolloutFK, costs1, a, i1, (i_update + 1) * 100, ts, dir1, goal1, n_samples);
                    i1++;
                }

                if (a == 2 && i2 < n_samples) {

                    cout << "Press enter to resume" << endl;
                    cin.ignore();
                    my_row = cost_vars2.row(i2);
                    rollout = (Map<MatrixXd>(my_row.data(), n_cost_vars, n_time_steps)).transpose();
                    ts = rollout.col(3 * n_dims);

                    for (int i = 0; i < ts.size(); i++) {
                        q1 = rollout(i, 0);
                        q2 = rollout(i, 1);
                        q3 = rollout(i, 2);
                        q4 = rollout(i, 3);
                        q5 = rollout(i, 4);
                        q6 = rollout(i, 5);

                        rolloutFKAvoidance(i, 0) = forwardKinematic(q1, q2, q3, q4, q5, q6, q7)(0, 3);
                        rolloutFKAvoidance(i, 1) = forwardKinematic(q1, q2, q3, q4, q5, q6, q7)(1, 3);
                        rolloutFKAvoidance(i, 2) = forwardKinematic(q1, q2, q3, q4, q5, q6, q7)(2, 3);
                    }

                    cout << "Press enter to start trajectory G2" << endl;
                    beginTime = ros::Time::now().toSec();

                    for (int hh = 0; hh < ts.size(); hh += sampleSize) {
                        msg_.points[0].positions[0] = rollout(hh, 0) ;
                        msg_.points[0].positions[1] = rollout(hh, 1) ;
                        msg_.points[0].positions[6] = rollout(hh, 2) ;
                        msg_.points[0].positions[2] = rollout(hh, 3) ;
                        msg_.points[0].positions[3] = rollout(hh, 4) ;
                        msg_.points[0].positions[4] = rollout(hh, 5) ;
                        msg_.points[0].positions[5] = rollout(hh, 6) ;

                        q1 = rollout(hh, 0);
                        q2 = rollout(hh, 1);
                        q3 = rollout(hh, 2);
                        q4 = rollout(hh, 3);
                        q5 = rollout(hh, 4);
                        q6 = rollout(hh, 5);

                        rolloutFK(hh, 0) = forwardKinematic(q1, q2, q3, q4, q5, q6, q7)(0, 3);
                        rolloutFK(hh, 1) = forwardKinematic(q1, q2, q3, q4, q5, q6, q7)(1, 3);
                        rolloutFK(hh, 2) = forwardKinematic(q1, q2, q3, q4, q5, q6, q7)(2, 3);

                        traj_pub.publish(msg_);
                        ros::spinOnce();
                        loop_rate.sleep();

                        if (ptompSet) {
                            returnNum = checkCollision(ts, rolloutFKAvoidance);

                            if (returnNum > 250)
                                returnNum = 250;
                            ptompSet = false;
                        }

                        if (hh == returnNum - 30) {
                            flag = false;
                        }
                        if (flag == true) {
                            hh = executeAvoidance(rollout, rolloutFK, hh, returnNum, msg_);
                            flag = false;
                            returnNum = -200;
                        }
                    }

                    std::string text;
                    std::stringstream ss;
                    ss << universalCount2;
                    string str = ss.str();
                    universalCount2++;

                    for (int hh = ts.size(); hh > 0; hh -= sampleSize) {
                        msg_.points[0].positions[0] = rollout(hh - 1, 0) ;
                        msg_.points[0].positions[1] = rollout(hh - 1, 1) ;
                        msg_.points[0].positions[6] = rollout(hh - 1, 2) ;
                        msg_.points[0].positions[2] = rollout(hh - 1, 3) ;
                        msg_.points[0].positions[3] = rollout(hh - 1, 4) ;
                        msg_.points[0].positions[4] = rollout(hh - 1, 5) ;
                        msg_.points[0].positions[5] = rollout(hh - 1, 6) ;

                        traj_pub.publish(msg_);
                        ros::spinOnce();
                        loop_rate.sleep();
                    }

                    if (predictedGoal == "0")
                        accuracy = 0;
                    else
                        accuracy = 1;

                    cout << "humans time is:" << human_time << endl;
                    cout << "accuracy is:" << accuracy << endl;
                    computeCosts(costs_detailed2, rollout, rolloutFK, costs2, a, i2, (i_update + 1) * 100, ts, dir2, goal2, n_samples);
                    i2++;
                }

                if (a == 3 && i3 < n_samples) {

                    cout << "Press enter to resume" << endl;
                    cin.ignore();
                    my_row = cost_vars3.row(i3);
                    rollout = (Map<MatrixXd>(my_row.data(), n_cost_vars, n_time_steps)).transpose();
                    ts = rollout.col(3 * n_dims);

                    for (int i = 0; i < ts.size(); i++) {
                        q1 = rollout(i, 0);
                        q2 = rollout(i, 1);
                        q3 = rollout(i, 2);
                        q4 = rollout(i, 3);
                        q5 = rollout(i, 4);
                        q6 = rollout(i, 5);

                        rolloutFKAvoidance(i, 0) = forwardKinematic(q1, q2, q3, q4, q5, q6, q7)(0, 3);
                        rolloutFKAvoidance(i, 1) = forwardKinematic(q1, q2, q3, q4, q5, q6, q7)(1, 3);
                        rolloutFKAvoidance(i, 2) = forwardKinematic(q1, q2, q3, q4, q5, q6, q7)(2, 3);
                    }

                    cout << "Press enter to start trajectory G3" << endl;
                    beginTime = ros::Time::now().toSec();

                    for (int hh = 0; hh < ts.size(); hh += sampleSize) {
                        msg_.points[0].positions[0] = rollout(hh, 0) ;
                        msg_.points[0].positions[1] = rollout(hh, 1) ;
                        msg_.points[0].positions[6] = rollout(hh, 2) ;
                        msg_.points[0].positions[2] = rollout(hh, 3) ;
                        msg_.points[0].positions[3] = rollout(hh, 4) ;
                        msg_.points[0].positions[4] = rollout(hh, 5) ;
                        msg_.points[0].positions[5] = rollout(hh, 6) ;

                        q1 = rollout(hh, 0);
                        q2 = rollout(hh, 1);
                        q3 = rollout(hh, 2);
                        q4 = rollout(hh, 3);
                        q5 = rollout(hh, 4);
                        q6 = rollout(hh, 5);

                        rolloutFK(hh, 0) = forwardKinematic(q1, q2, q3, q4, q5, q6, q7)(0, 3);
                        rolloutFK(hh, 1) = forwardKinematic(q1, q2, q3, q4, q5, q6, q7)(1, 3);
                        rolloutFK(hh, 2) = forwardKinematic(q1, q2, q3, q4, q5, q6, q7)(2, 3);

                        traj_pub.publish(msg_);
                        ros::spinOnce();
                        loop_rate.sleep();

                        if (ptompSet) {
                            returnNum = checkCollision(ts, rolloutFKAvoidance);

                            if (returnNum > 250)
                                returnNum = 250;
                            ptompSet = false;
                        }

                        if (hh == returnNum - 30) {
                            flag = false;
                        }
                        if (flag == true) {
                            hh = executeAvoidance(rollout, rolloutFK, hh, returnNum, msg_);
                            flag = false;
                            returnNum = -200;
                        }
                    }

                    std::string text;
                    std::stringstream ss;
                    ss << universalCount3;
                    string str = ss.str();
                    universalCount3++;

                    for (int hh = ts.size(); hh > 0; hh -= sampleSize) {
                        msg_.points[0].positions[0] = rollout(hh - 1, 0) ;
                        msg_.points[0].positions[1] = rollout(hh - 1, 1) ;
                        msg_.points[0].positions[6] = rollout(hh - 1, 2) ;
                        msg_.points[0].positions[2] = rollout(hh - 1, 3) ;
                        msg_.points[0].positions[3] = rollout(hh - 1, 4) ;
                        msg_.points[0].positions[4] = rollout(hh - 1, 5) ;
                        msg_.points[0].positions[5] = rollout(hh - 1, 6) ;

                        traj_pub.publish(msg_);
                        ros::spinOnce();
                        loop_rate.sleep();
                    }

                    if (predictedGoal == "2")
                        accuracy = 0;
                    else
                        accuracy = 1;

                    cout << "humans time is:" << human_time << endl;
                    cout << "accuracy is:" << accuracy << endl;
                    computeCosts(costs_detailed3, rollout, rolloutFK, costs3, a, i3, (i_update + 1) * 100, ts, dir3, goal3, n_samples);
                    i3++;
                }
            }

            std::cout << "costs 1 = " << costs1.format(CleanFmt) << endl;
            std::cout << "costs 2 = " << costs2.format(CleanFmt) << endl;
            std::cout << "costs 3 = " << costs3.format(CleanFmt) << endl;
        }

        // 3. Update parameters
        updater1->updateDistribution(*distribution1, samples1, costs1, *distribution1, update_summary1);
        updater2->updateDistribution(*distribution2, samples2, costs2, *distribution2, update_summary2);
        updater3->updateDistribution(*distribution3, samples3, costs3, *distribution3, update_summary3);

        update_summary1.cost_eval = cost_eval1[0];
        update_summary1.cost_vars_eval = cost_vars_eval1;
        update_summary1.cost_vars = cost_vars1;
        update_summaries1.push_back(update_summary1);

        update_summary2.cost_eval = cost_eval2[0];
        update_summary2.cost_vars_eval = cost_vars_eval2;
        update_summary2.cost_vars = cost_vars2;
        update_summaries2.push_back(update_summary2);

        update_summary3.cost_eval = cost_eval3[0];
        update_summary3.cost_vars_eval = cost_vars_eval3;
        update_summary3.cost_vars = cost_vars3;
        update_summaries3.push_back(update_summary3);
    }
}

void getPtomp(const sensor_msgs::JointState::ConstPtr& msg)
{

    predictedGoal = msg->name[0];
    for (int i = 0; i < 51; i++) {
        ptompData(i, 0) = msg->position[i];
        ptompData(i, 1) = msg->velocity[i];
        ptompData(i, 2) = msg->effort[i];
        ptompSet = true;
    }
}

void getTime(const std_msgs::Float32::ConstPtr& msg)
{
    int64_t end = ros::Time::now().toSec();
    human_time = (end - beginTime);
    trust = msg->data;
}

void computeCosts(Eigen::MatrixXd& costs_detailed, Eigen::MatrixXd& rollout, Eigen::MatrixXd& rolloutFK, Eigen::VectorXd& costs, int a, int ii, double i_update, Eigen::VectorXd& ts, string dir,
    VectorXd& goal, int n_samples)
{

    int n_time_steps = rollout.rows();
    int n_dims = goal.size();
    std::stringstream ss;
    ss << i_update;
    string str = ss.str();

    std::stringstream ss2;
    ss2 << a;
    string str2 = ss2.str();

    std::string text;
    MatrixXd rolloutFKd(n_time_steps - 1, 3);
    MatrixXd rolloutFKdd(rolloutFKd.rows() - 1, 3);

    //Angular Jerk
    ydd = rollout.block(0, 2 * n_dims, n_time_steps, n_dims);
    time_step = ts[1] - ts[0];
    if (jerk_weight_ != 0.0) {
        jerk_sum = 0;
        for (int i = 0; i < ts.size() - 1; i++) {
            jerk_sum += abs(ydd(i + 1, 0) - ydd(i, 0)) / time_step;
            jerk_sum += abs(ydd(i + 1, 1) - ydd(i, 1)) / time_step;
            jerk_sum += abs(ydd(i + 1, 2) - ydd(i, 2)) / time_step;
        }
    }

    //FK Derivatives Calculations

    for (int i = 0; i < ts.size() - 1; i++) {
        rolloutFKd(i, 0) = (rolloutFK(i + 1, 0) - rolloutFK(i, 0)) / time_step;
        rolloutFKd(i, 1) = (rolloutFK(i + 1, 1) - rolloutFK(i, 1)) / time_step;
        rolloutFKd(i, 2) = (rolloutFK(i + 1, 2) - rolloutFK(i, 2)) / time_step;
    }
    for (int i = 0; i < rolloutFKd.rows() - 1; i++) {
        rolloutFKdd(i, 0) = (rolloutFKd(i + 1, 0) - rolloutFKd(i, 0)) / time_step;
        rolloutFKdd(i, 1) = (rolloutFKd(i + 1, 1) - rolloutFKd(i, 1)) / time_step;
        rolloutFKdd(i, 2) = (rolloutFKd(i + 1, 2) - rolloutFKd(i, 2)) / time_step;
    }

    //EndEffector Jerk
    if (jerk_weight_2_ != 0.0) {
        jerk_sum_2 = 0;
        for (int i = 0; i < rolloutFKdd.rows() - 1; i++) {
            jerk_sum_2 += abs(rolloutFKdd(i + 1, 0) - rolloutFKdd(i, 0)) / time_step;
            jerk_sum_2 += abs(rolloutFKdd(i + 1, 1) - rolloutFKdd(i, 1)) / time_step;
            jerk_sum_2 += abs(rolloutFKdd(i + 1, 2) - rolloutFKdd(i, 2)) / time_step;
        }
    }

    costs[ii] = human_time_weight_ * human_time + accuracy_weight_ * accuracy + jerk_weight_ * (jerk_sum - 1500) / 1000 + jerk_weight_2_ * (jerk_sum_2 - 1200) / 1500 + obstacle_avoidance_weight_ * obstacle_cost + trust * trust_region_weight_;

    costs_detailed(ii, 0) = human_time;
    costs_detailed(ii, 1) = robot_time;
    costs_detailed(ii, 2) = accuracy;
    costs_detailed(ii, 3) = jerk_sum;
    costs_detailed(ii, 4) = jerk_sum_2;
    costs_detailed(ii, 5) = human_time_weight_ * human_time;
    costs_detailed(ii, 6) = accuracy_weight_ * accuracy;
    costs_detailed(ii, 7) = (jerk_sum - 1500) / 1000;
    costs_detailed(ii, 8) = (jerk_sum_2 - 1200) / 1500;
    costs_detailed(ii, 9) = obstacle_avoidance_weight_ * obstacle_cost;
    costs_detailed(ii, 10) = trust * trust_region_weight_;
    costs_detailed(ii, 11) = costs[ii];
}

Eigen::MatrixXd forwardKinematic(double q1, double q2, double q3, double q4, double q5, double q6, double q7)
{
    MatrixXd aOne(4, 4), aTwo(4, 4), aThree(4, 4), aFour(4, 4), aFive(4, 4), aSix(4, 4), aSeven(4, 4), resultFinal(4, 4);

    aOne << cos(q1), 0, sin(q1), 0,
        sin(q1), 0, -1 * cos(q1), 0,
        0, 1, 0, 0,
        0, 0, 0, 1;

    aTwo << cos(q2), 0, -1 * sin(q2), 0,
        sin(q2), 0, cos(q2), 0,
        0, -1, 0, 0,
        0, 0, 0, 1;

    aThree << cos(q3), 0, -1 * sin(q3), 0,
        sin(q3), 0, cos(q3), 0,
        0, -1, 0, 0.4,
        0, 0, 0, 1;

    aFour << cos(q4), 0, sin(q4), 0,
        sin(q4), 0, -1 * cos(q4), 0,
        0, 1, 0, 0,
        0, 0, 0, 1;

    aFive << cos(q5), 0, sin(q5), 0,
        sin(q5), 0, -1 * cos(q5), 0,
        0, 1, 0, 0.39,
        0, 0, 0, 1;

    aSix << cos(q6), 0, -1 * sin(q6), 0,
        sin(q6), 0, cos(q6), 0,
        0, -1, 0, 0,
        0, 0, 0, 1;

    aSeven << cos(q7), -1 * sin(q7), 0, 0,
        sin(q7), cos(q7), 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1;

    resultFinal = aOne * aTwo * aThree * aFour * aFive * aSix * aSeven;
    resultFinal(1, 3) = -1 * (resultFinal(1, 3));
    resultFinal(2, 3) = -1 * (resultFinal(2, 3) + 0.3105);
    return resultFinal;
}

int checkCollision(Eigen::MatrixXd ts, Eigen::MatrixXd rolloutFKAvoidance)
{
    RowVector3d position_one, position_two, distanceVec;
    double distance;
    double min = 5.0;
    int num = 0;
    MatrixXd ptomp = ptompData;

    for (int i = 0; i < ptomp.rows(); i++) {
        position_one = ptomp.row(i);
        for (int j = 0; j < ts.size(); j++) {
            position_two = rolloutFKAvoidance.row(j);
            distanceVec = position_two - position_one;
            distance = distanceVec.norm();
            if (distance <= min) {
                num = j;
                min = distance;
            }
        }
    }
}

int executeAvoidance(MatrixXd rollout, MatrixXd rolloutFK, int hh, int returnNum, trajectory_msgs::JointTrajectory msg_)
{

    double x1 = rollout(hh, 1);
    double y, z, w;
    z = rollout(returnNum + 35, 1);
    y = 0.01;
    for (int i = hh + 1; i < returnNum; i++) {
        msg_.points[0].positions[0] = rollout(i, 0) ;
        msg_.points[0].positions[1] = x1-y ;
        msg_.points[0].positions[6] = rollout(i, 2) ;
        msg_.points[0].positions[2] = rollout(i, 3) ;
        msg_.points[0].positions[3] = rollout(i, 4) ;
        msg_.points[0].positions[4] = rollout(i, 5) ;
        msg_.points[0].positions[5] = rollout(i, 6) ;
        traj_pub.publish(msg_);
        ros::spinOnce();
        //loop_rate.sleep();
        std::this_thread::sleep_for(avoiding);
        x1 = x1 - y;
    }
    w = z - x1;
    for (int i = returnNum; i < returnNum + 35; i++) {
        msg_.points[0].positions[0] = rollout(i, 0) ;
        msg_.points[0].positions[1] = x1 + (w / 35) ;
        msg_.points[0].positions[6] = rollout(i, 2) ;
        msg_.points[0].positions[2] = rollout(i, 3) ;
        msg_.points[0].positions[3] = rollout(i, 4) ;
        msg_.points[0].positions[4] = rollout(i, 5) ;
        msg_.points[0].positions[5] = rollout(i, 6) ;
        traj_pub.publish(msg_);
        ros::spinOnce();
        //loop_rate.sleep();
        std::this_thread::sleep_for(avoiding);
        x1 = x1 + (w / 35);
    }
    flag = false;
    return returnNum + 35;
}
