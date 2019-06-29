// this program should publish messages to control the robot using dmp in combination with inverse kinematics 
#include <string>
#include <set>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/QR>

#include "/home/hrcstudents/Documents/Code/catkin_ws/src/dmp_bbo_node/include/dmp_bbo_node/TaskSolver.hpp"
#include "/home/hrcstudents/Documents/Code/catkin_ws/src/dmp_bbo_node/include/dmp_bbo_node/TaskSolverDmp.hpp"
#include "/home/hrcstudents/Documents/Code/catkin_ws/src/dmp_bbo_node/include/dmp_bbo_node/Task.hpp"
#include "/home/hrcstudents/Documents/Code/catkin_ws/src/dmp_bbo_node/include/dmp_bbo_node/TaskViapoint.hpp"
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
#include "/home/hrcstudents/Documents/Code/catkin_ws/src/dmp_bbo_node/include/dmp_bbo_node/runEvolutionaryOptimization.hpp"
#include "ros/ros.h"
#include "std_msgs/String.h"
#include "std_msgs/Float32.h"
#include "std_msgs/Bool.h"
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
//#include "MatlabEngine.hpp"
//#include "MatlabDataArray.hpp"
using namespace std;
using namespace Eigen;
using namespace DmpBbo;

//questions: do i have to incorporate the repellent force into the dmp?

const double PI = 3.141592653589793238463;

//global variables, get filled by callback function from subscriber
Vector3d obstaclePosition = Vector3d::Zero();
Vector3d closestJointPosition = Vector3d::Zero();
bool collision = false;			//gets used in computeCosts

//function to publish trajectory messages to the robot, implemented below main()
void publishMessages(VectorXd& ts, MatrixXd& ys, MatrixXd& yds, MatrixXd& ydds, ros::Publisher& trajPub, ros::Rate loop_rate, int nDims);


//function takes costVariables and computes cost, implemented below
void computeCosts(MatrixXd& ys, MatrixXd& yds, MatrixXd& ydds, int nDims, VectorXd& cost, VectorXd& ts);

//computes pseudo inverse for the jacobian, needed for the angularVelocities, impelemted below
MatrixXd getPseudoInverse(MatrixXd& J);

//computes the angular velocities for the joints through inverse kinematics with null space constraints, implemented below
MatrixXd getAngularVelocities(MatrixXd& ys, MatrixXd& yds, VectorXd& ts);

//function computes forward kinematics using ys which includes all joint angles, implemented below
MatrixXd getEndEffectorPosition(MatrixXd& ys, VectorXd& ts);

//function computes end effector velocity from joint velocities, uses getJacobian() function, implemented below
MatrixXd getEndEffectorVelocity(MatrixXd& ys, MatrixXd& yds, VectorXd& ts);

//function to get the Jacobian, implemented below, only used for the getEndEffectorVelocity function
MatrixXd getJacobian(double q1, double q2, double q3, double q4, double q5, double q6, double q7);

//computes the forward kinematics for the getEndEffectorVelocity() function
MatrixXd forwardKinematics(double q1, double q2, double q3, double q4, double q5, double q6, double q7);

//gradient calculations for potential force
Vector3d getGradientCos(Vector3d& vel, Vector3d& pos, double dist) {
	Vector3d gradient;
	double den = vel.norm()*pow(dist,2);	//denumerator
	Vector3d term1 = vel*vel.norm()*dist;
	Vector3d term2 = (double)(vel.transpose()*pos)*vel.norm()/dist*pos;
	gradient = (term1 - term2)/den; 
	return gradient;
}

Vector3d getGradientDistance(double dist, Vector3d& pos) {
	double factor = 1/dist;
	return factor*pos;
}

//function to compute the repellent force resulting from the dynamic potential field for the end effector
//it needs xds: velocity of end effector, xs: position of end effector, x_obs: position of obstacle, xd_obs: velocity of obstacle
MatrixXd dynamicPotentialForce(MatrixXd& ys, MatrixXd& yds/*, VectorXd x_obs = VectorXd::Zero(3), VectorXd xd_obs = VectorXd::Zero(3)*/, int nDims, VectorXd& ts) {
	double lambda = 1, beta = 1;  					//parameters to include in policy improvement
	MatrixXd force(ts.size(), 3);					//force in every direction for all times
	MatrixXd torque(ts.size(), 7), jacobian(3,7); 	//force applied to the joints, transform with jacobian below

	Vector3d x_obs = obstaclePosition, xd_obs = Vector3d::Zero();								//only temporary because we don't have obstacle information yet


	double q1 = ys(0), q2 = ys(1), q3 = ys(2), q4 = ys(3), q5 = ys(4), q6 = ys(5), q7 = ys(6);	//for computation of jacobian

	for(int t = 0; t<ts.size(); t++) {			//we have the end effector postition and the jacobian for all times
		Vector3d pos = getEndEffectorPosition(ys, ts).row(t);		//better to use here the closest point to obstacle
		double dist = (pos - x_obs).norm();		//distance between obstacle and end effector p(x)

		Vector3d vel = getEndEffectorVelocity(ys, yds, ts).row(t);
		Vector3d rel_vel = vel - xd_obs;	//relative velocity
		//the following calculation was adapted from the dynamic potential field equations presented in Hoffmann's paper
		double temp = rel_vel.norm()*dist;
		double cos_theta = (double)(rel_vel.transpose()*pos)/temp;	
		VectorXd ftemp(3), ttemp(7);			//temporary force, temporary torque
		if(acos(cos_theta) > PI/2 && acos(cos_theta) <= PI) {
			ftemp = lambda * pow(-cos_theta, beta-1) * rel_vel.norm()/dist * (beta*getGradientCos(vel, pos, dist) - cos_theta/dist*getGradientDistance(dist, pos));
		}	
		else {
			ftemp = Vector3d::Zero();
		}
		force(t,0) = ftemp[0];
		force(t,1) = ftemp[1];
		force(t,2) = ftemp[2];
		//compute torque with jacobian
		q1 = ys(t,0); q2 = ys(t,1); q3 = ys(t,2); q4 = ys(t,3); q5 = ys(t,4); q6 = ys(t,5); q7 = ys(t,6); 	//compute new jacobian in each time step
		jacobian = getJacobian(q1,q2,q3,q4,q5,q6,q7);
		ttemp = ftemp.transpose()*getPseudoInverse(jacobian).transpose();
		for(int n = 0; n < 7; n++) {
			torque(t,n) = ttemp(n);
		}
	}

	return torque;
}

//callback function for obsSub, saves info in global variable obstaclePosition
void getObstaclePosition(geometry_msgs::Point point) {
	obstaclePosition[0] = point.x;
	obstaclePosition[1] = point.y;
	obstaclePosition[2] = point.z;
}

void getClosestJoint(geometry_msgs::Point point) {
	closestJointPosition[0] = point.x;
	closestJointPosition[1] = point.y;
	closestJointPosition[2] = point.z;
}

void setCollisionStatus(std_msgs::Bool status) {
	collision = status.data;
	//cout << "collision: " << collision << endl;
}

int main(int argc, char *argv[])
{
	//define node
	ros::init(argc, argv, "dmp_ik");
	ros::NodeHandle node;

	//define publisher, Robot Publisher
	string trajTopic("/joint_trajectory_controller/command");
	ros::Publisher trajPub = node.advertise<trajectory_msgs::JointTrajectory>(trajTopic.c_str(), 4000);		
	ros::Rate loop_rate(100);
	//define subscriber to get obstacle position and closestPointToObstacle position from vrep
	ros::Subscriber obsSub = node.subscribe("/position/obstacle", 100, getObstaclePosition);
	ros::Subscriber jointSub = node.subscribe("position/closestJoint", 100, getClosestJoint);
	//define subscriber for collision status from vrep
	ros::Subscriber collSub = node.subscribe("/collision_status", 1000, setCollisionStatus);
	//define subscriber(s) for Ptomp, and time
	
	//define goal and start
	VectorXd start(7); start << 20 * PI / 180, 50 * PI / 180, 50 * PI / 180, -60 * PI / 180, 0 * PI / 180, 0, 0;	//initial state
	//VectorXd goal (7); goal  << 40 * PI / 180, 70 * PI / 180, 50 * PI / 180, -95 * PI / 180, 0 * PI / 180, 0, 0;	//attractor state
	//VectorXd goal (7); goal  << 0 , 0, 0 , 0 , 0 , 0, 0;
	//VectorXd goal(7); goal << -60 * PI / 180.0, 90 * PI / 180.0, 70 * PI / 180.0, -75 * PI / 180.0, 0 * PI / 180.0, 0 * PI / 180.0, 0;
	VectorXd goal(7); goal << 35 * PI / 180.0, -80 * PI / 180.0, 60 * PI / 180.0, -70 * PI / 180.0, 0 * PI / 180.0, 0 * PI / 180.0, 0;		//goes to blue

	///////////////////////////////////////////////////define the DMP///////////////////////////////////////////////////////////////////////////////////////////////
	double tau = 2.5;		//time constant
	double alphaGoal = 15;	//decay constant
 
 	VectorXd one  = VectorXd::Ones(1);
 	VectorXd zero = VectorXd::Zero(1);
	DynamicalSystem* goalSys   = new ExponentialSystem(tau, start, goal, alphaGoal);	//prevents high accelerations in the beginning
	DynamicalSystem* phaseSys  = new TimeSystem(tau);									//ensures autonomy of forcing term, 1D
	DynamicalSystem* gatingSys = new ExponentialSystem(tau, one, zero, 5);				//starts at one, ends at 0, decay const = 5, ensures convergence, 1D

	double dt = 0.01;											//integration step duration
	int timeSteps = ceil(tau/dt)+1;								//number of time steps 
	VectorXd ts = VectorXd::LinSpaced(timeSteps, 0.0, tau);		//vector with all time steps

	Trajectory trajectory = Trajectory::generateMinJerkTrajectory(ts, start, goal);

	int nBasisFunc = 20;
	int nDims = 7; //dimension of goal (7 joints)
	//double intersectionHeight = 0.9; default is 0.5
	MetaParametersRBFN* metaParam = new MetaParametersRBFN(1, nBasisFunc); //1 is the expected input dimension
	
	//we need a vector with as much function approximators (basis functions) as the dimensionality of the DMP
	vector<FunctionApproximator*> funcAppr(nDims);
	for(int iDim = 0; iDim < nDims; iDim ++) {
		funcAppr[iDim] = new FunctionApproximatorRBFN(metaParam);
	} 

	double alphaSpringDamper = 20;  //value taken from old code
	Dmp* dmp = new Dmp(nDims, funcAppr, alphaSpringDamper, goalSys, phaseSys, gatingSys);
	string directory = "/home/hrcstudents/Documents/Code/catkin_ws/ResultsUpdate/1D";
	dmp->train(trajectory, directory, true);		//DMP learns the trajectory and saves the result in the directory, true -> overwrite

	set<string> parameters_to_optimize;
    parameters_to_optimize.insert("weights");		//we optimize the weights of the RBFN
    double integrate_dmp_beyond_tau_factor = 1.2; 	//value taken from old code
    TaskSolverDmp* taskSolver = new TaskSolverDmp(dmp, parameters_to_optimize, dt, integrate_dmp_beyond_tau_factor);

   
    //Analytical solution of our system containing all the information we need, taken from old code, for plotting (?)
    MatrixXd xs_ana;	//state matrix
    MatrixXd xds_ana;	//rates of change (-> derivative??)
    MatrixXd forcing_terms_ana, fa_output_ana;	//those two are for debugging purposes only
    dmp->analyticalSolution(ts, xs_ana, xds_ana, forcing_terms_ana, fa_output_ana);
    MatrixXd output_ana(ts.size(), 1 + 2*dmp->dim());		//1 for time steps + dim(xs) + dim(xds), usually dim(xs)=dim(xds), eventually use dmp->dim()
    output_ana << xs_ana, xds_ana, ts;
    bool overwrite = true;
    saveMatrix(directory, "reproduced_xs_xds1.txt", output_ana, overwrite);
    saveMatrix(directory, "reproduced_forcing_terms1.txt", forcing_terms_ana, overwrite);
    saveMatrix(directory, "reproduced_fa_output1.txt", fa_output_ana, overwrite);
    //until here just creating the dmp
	
    /*		//is defined in the beginning
    //define task, viapoint is an example, probably implement a new subclass of task with our needs, this one cannot evaluate rollouts...
    VectorXd viapoint(3); viapoint << 1, 5, 10;		//chosen arbitrarily
    double viapointTime = 0.3;						//also arbitrarily, time at which the viapoint has to be passed
    TaskViapoint* task = new TaskViapoint(viapoint, viapointTime);
	*/
    /////////////////////////////////////////////////////run the policy improvement/////////////////////////////////////////////////////////////////////////////////
    //Make the initial distribution for exploration
    VectorXd meanInit;
    dmp->getParameterVectorSelected(meanInit);	//-> centers around the weights
    double covarSize = 1000; 					//the bigger the variance, the bigger the sample space
    MatrixXd covarInit = covarSize * MatrixXd::Identity(meanInit.size(), meanInit.size());
    DistributionGaussian* distribution = new DistributionGaussian(meanInit, covarInit);		//distribution around the parameters to sample
	
	// Make the parameter updater
    double eliteness = 10;				//values from old code
    double covarDecayFactor = 0.6;		//sample space gets smaller with time
    string weightingMethod("PI-BB");	//BB: reward-weighted averaging
    Updater* updater = new UpdaterCovarDecay(eliteness, covarDecayFactor, weightingMethod);

    //Run the optimization
    int nUpdates = 100;		//change
    int nSamplesPerUpdate = 10;
    //string saveDirectory = "/home/hrcstudents/Documents/Code/catkin_ws/ResultsUpdate";
    //runEvolutionaryOptimization(task, taskSolver, distribution, updater, nUpdates, nSamplesPerUpdate, saveDirectory, true);	//this function creates segmentation fault :(
    
    VectorXd costs(10);		//error when less that 10 -> weird
    // 0. Get cost of current distribution mean
    MatrixXd samples, costVariables;
    for(int iUpdate = 0; iUpdate < nUpdates; iUpdate++) {
    	cout << "Update Nr: " << iUpdate << endl;
    	distribution->generateSamples(nSamplesPerUpdate, samples);		//you generate multiple samples
    	taskSolver->performRollouts(samples, costVariables);			//costVariables now includes y_out, yd_out, ydd_out, ts, forcing terms

		MatrixXd ys_ana, yds_ana, ydds_ana;								//the trajectory values that will be published
		dmp->analyticalSolution(ts, xs_ana, xds_ana, forcing_terms_ana, fa_output_ana);
		dmp->statesAsTrajectory(xs_ana, xds_ana, ys_ana, yds_ana, ydds_ana);		//convert states to trajectory values, ys: joint angles, yds: joint velocities, ydds: joint accelerations

    	computeCosts(ys_ana, yds_ana, ydds_ana, nDims, costs, ts);									//computes cost

    	//obstacle avoidance for end effector
    	MatrixXd torque = dynamicPotentialForce(ys_ana, yds_ana, nDims, ts)/tau;		//add this to the published velocities (right?)		
    	ydds_ana = ydds_ana + torque;
    	///////////////////////////////////////////////////////////////write and publish message///////////////////////////////////////////////////////////////////////////////////
		publishMessages(ts, ys_ana, yds_ana, ydds_ana, trajPub, loop_rate, nDims);		//add here the potential force/tau 

    	updater->updateDistribution(*distribution, samples, costs, *distribution);

    	for(int i = 0; i < costs.size(); i++) {		//for testing
			cout << "costs: " << costs[i] << ", " << flush;
    	}
    	cout << "\n" << endl;
    } 
    
    //cout << "costVariables.size(): " << costVariables.size() << endl;
    //cout << "nSamples: " << nSamples << endl;
    //cout << "xs_ana.cols(): " << xs_ana.cols() << ", xds_ana.cols(): " << xds_ana.cols() << ", dmp.dim(): " << dmp->dim() << endl;

    //test
    
    //int nTimeSteps = costVar.rows();
    //VectorXd xs = costVar.block(0, 0, nTimeSteps, nDims);
    //VectorXd xds = costVar.block(0, nDims, nTimeSteps, nDims);
    

	//TODO: destroy objects
	return 0;
}

void publishMessages(VectorXd& ts, MatrixXd& ys, MatrixXd& yds, MatrixXd& ydds, ros::Publisher& trajPub, ros::Rate loop_rate, int nDims) {	
	
	trajectory_msgs::JointTrajectory msg;

    msg.joint_names.clear();
    msg.joint_names.push_back("a1_joint");
    msg.joint_names.push_back("a2_joint");
    msg.joint_names.push_back("a3_joint");
    msg.joint_names.push_back("a4_joint");
    msg.joint_names.push_back("a5_joint");
    msg.joint_names.push_back("a6_joint");
    msg.joint_names.push_back("e1_joint");
    msg.points.resize(1);						//one point with 7 positions, 7 velocities and 7 accelerations for 7 joints
    msg.points[0].positions.resize(7); 
    msg.points[0].velocities.resize(7);
    msg.points[0].accelerations.resize(7);
    //msg.points[0].time_from_start = ros::Duration(0.015*sampleSize);

	//MatrixXd costVar = costVariables.transpose();							//so that the rows are the time steps and the columns are y, yd, ydd, ts, ft
	//cout << "ys.rows(): " << ys.rows() << ", ys.cols(): " << ys.cols() << ", ts.size(): " << ts.size() << endl;		
	for(int i = 0; i < ts.size(); i++) {	//for all times
		for(int j = 0; j < nDims; j++) {	//for all joints
			msg.points[0].positions[j] = ys(i, j);
			msg.points[0].velocities[j] = yds(i, j);		//for the future: only th end effector will receive signals from DMP, the rest from inverse kinematics
			msg.points[0].accelerations[j] = ydds(i, j);
		}
		trajPub.publish(msg);
		ros::spinOnce();				//need if we have a subscribtion in the program
		loop_rate.sleep();
	}
}

void computeCosts(MatrixXd& ys, MatrixXd& yds, MatrixXd& ydds, int nDims, VectorXd& cost, VectorXd& ts) {
	//int nTimeSteps = ys.rows();
	
	double wHuman = 1, wRobot = 1, wAccuracy = 1, wJoint = 0.01, wEndeffector = 1, wCollision = 1000;		//weights to make all variables in the same range
	double human_time = 1, robot_time = 1, accuracy = 0, joint_jerk = 1, endeffector_jerk = 0;
	
	for(int i = 0; i < cost.size(); i++) {
		cost[i] = 0;
	}

	//calculate angular jerk
	double deltaT = ts[1] - ts[0];
	for(int i = 0; i < ts.size()-1; i++) {
		for(int j = 0; j < nDims; j++) {
			joint_jerk += abs(ydds(i+1,j) - ydds(i,j))/deltaT;		//derive acceleration
		}
	}

	cout << "collision: " << collision << endl;

	cost[1] = human_time*wHuman;		
	cost[2] = robot_time*wRobot;
	cost[3] = accuracy*wAccuracy;
	cost[4] = joint_jerk*wJoint;
	cost[5] = endeffector_jerk*wEndeffector;
	cost[6] = wCollision*collision;

	cost[0] = cost.sum();

}

MatrixXd forwardKinematics(double q1, double q2, double q3, double q4, double q5, double q6, double q7) {
	MatrixXd aOne(4, 4), aTwo(4, 4), aThree(4, 4), aFour(4, 4), aFive(4, 4), aSix(4, 4), aSeven(4, 4), transformMat(4, 4);

	aOne << cos(q1), 0, sin(q1), 0,				//the lengths of the links have to be added..
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
        	sin(q6), 0, cos(q6)		, 0,
        	0      ,-1, 0			, 0,
        	0	   , 0, 0			, 1;

    aSeven << cos(q7), -1 * sin(q7), 0, 0,
        	sin(q7), cos(q7), 0, 0,
        	0, 0, 1, 0,
        	0, 0, 0, 1;

   	transformMat = aOne * aTwo * aThree * aFour * aFive * aSix * aSeven;
   	return transformMat;
}

MatrixXd getEndEffectorPosition(MatrixXd& ys, VectorXd& ts) {
	MatrixXd transformMat(4,4), position(ts.size(), 3);	//x,y,z postion for all times
	transformMat = Matrix4d::Zero();
	double q1 = ys(0), q2 = ys(1), q3 = ys(2), q4 = ys(3), q5 = ys(4), q6 = ys(5), q7 = ys(6);

	for(int t = 0; t < ts.size(); t++) {
		q1 = ys(t,0); q2 = ys(t,1); q3 = ys(t,2); q4 = ys(t,3); q5 = ys(t,4); q6 = ys(t,5); q7 = ys(t,6);

	   	transformMat = forwardKinematics(q1, q2, q3, q4, q5, q6, q7);
	   	position(t,0) = transformMat(0,3);  //take only the translation part of the transformation matrix
	   	position(t,1) = transformMat(1,3);
	   	position(t,2) = transformMat(2,3);
	}
   	return position;
}

MatrixXd getJacobian(double q1, double q2, double q3, double q4, double q5, double q6, double q7) {
	MatrixXd jacobian(3,7); //3 cartesian dimensions, 7 joints
	//the following jacobian was computed symbolically with MATLAB with the forward kinematics in the getEndEffectorPosition function
	jacobian(0,0) = (2*sin(q1)*sin(q2))/5 - (39*sin(q5)*(cos(q1)*sin(q3) + cos(q2)*cos(q3)*sin(q1)))/100 + (39*cos(q4)*sin(q1)*sin(q2))/100;
	jacobian(0,1) = - (2*cos(q1)*cos(q2))/5 - (39*cos(q1)*cos(q2)*cos(q4))/100 - (39*cos(q1)*cos(q3)*sin(q2)*sin(q5))/100;
	jacobian(0,2) = -(39*sin(q5)*(cos(q3)*sin(q1) + cos(q1)*cos(q2)*sin(q3)))/100;
	jacobian(0,3) = (39*cos(q1)*sin(q2)*sin(q4))/100;
	jacobian(0,4) = (39*cos(q1)*sin(q2)*sin(q4))/100;
	jacobian(0,5) = 0;
	jacobian(0,6) = 0;

	jacobian(1,0) = - (2*cos(q1)*sin(q2))/5 - (39*sin(q5)*(sin(q1)*sin(q3) - cos(q1)*cos(q2)*cos(q3)))/100 - (39*cos(q1)*cos(q4)*sin(q2))/100;
	jacobian(1,1) = - (2*cos(q2)*sin(q1))/5 - (39*cos(q2)*cos(q4)*sin(q1))/100 - (39*cos(q3)*sin(q1)*sin(q2)*sin(q5))/100;
	jacobian(1,2) = (39*sin(q5)*(cos(q1)*cos(q3) - cos(q2)*sin(q1)*sin(q3)))/100;
	jacobian(1,3) = (39*sin(q1)*sin(q2)*sin(q4))/100;
	jacobian(1,4) = (39*cos(q5)*(cos(q1)*sin(q3) + cos(q2)*cos(q3)*sin(q1)))/100;
	jacobian(1,5) = 0;
	jacobian(1,6) = 0;

	jacobian(2,0) = 0;
	jacobian(2,1) = (39*cos(q2)*cos(q3)*sin(q5))/100 - (39*cos(q4)*sin(q2))/100 - (2*sin(q2))/5;
	jacobian(2,2) = -(39*sin(q2)*sin(q3)*sin(q5))/100;
	jacobian(2,3) = -(39*cos(q2)*sin(q4))/100;
	jacobian(2,4) = (39*cos(q3)*cos(q5)*sin(q2))/100;
	jacobian(2,5) = 0;
	jacobian(2,6) = 0;

	return jacobian;
}

MatrixXd getEndEffectorVelocity(MatrixXd& ys, MatrixXd& yds, VectorXd& ts) {
	double q1 = ys(0), q2 = ys(1), q3 = ys(2), q4 = ys(3), q5 = ys(4), q6 = ys(5), q7 = ys(6);
	MatrixXd jacobian(3,7);  	//jacobian has 3 rows (3 cartesian dimenstions) and 7 rows (7 joints)
	MatrixXd velocity(ts.size(),3);		//x,y,z velocity for all times
	VectorXd temp;                      //just to store temporary values in for loop

	for(int t = 0; t < ts.size(); t++) {
		q1 = ys(t,0); q2 = ys(t,1); q3 = ys(t,2); q4 = ys(t,3); q5 = ys(t,4); q6 = ys(t,5); q7 = ys(t,6);
		jacobian = getJacobian(q1, q2, q3, q4, q5, q6, q7);
		temp = jacobian*yds.row(t).transpose();
		velocity(t,0) = temp[0];
		velocity(t,1) = temp[1];
		velocity(t,2) = temp[2];
	}
	return velocity;
}

MatrixXd getPseudoInverse(MatrixXd& J) {
	MatrixXd temp = J.transpose()*J;
	return temp.inverse()*J.transpose();
}

MatrixXd getAngularVelocities(MatrixXd& ys, MatrixXd& yds, VectorXd& ts) {
	double q1 = ys(0), q2 = ys(1), q3 = ys(2), q4 = ys(3), q5 = ys(4), q6 = ys(5), q7 = ys(6);
	MatrixXd thetaDot(ts.size(),7);	//7 joint angle velocities for all times
	MatrixXd J(3,7), Jp(7,3);				//Jacobian, 3 dims x 7 joints, pseudo inverse
	for(int t = 0; t < ts.size(); t++) {
		q1 = ys(t,0); q2 = ys(t,1); q3 = ys(t,2); q4 = ys(t,3); q5 = ys(t,4); q6 = ys(t,5); q7 = ys(t,6);
		J = getJacobian(q1, q2, q3, q4, q5, q6, q7);
		Jp = getPseudoInverse(J);
		
	}
	return thetaDot;
}
/*
//the DH parameters have been aquired by inspection according to the DH norm
	aOne = transformation(0.3, 0,    0, q[0]);

    aTwo = transformation(0.2,  -PI/2, 0, q[1]);

    aThree = transformation(0.2, PI/2, 0, q[2]);

    aFour = transformation(0.2,   PI/2, 0, q[3]);

    aFive = transformation(0.2,-PI/2, 0, q[4]);

    aSix = transformation(0.19,  -PI/2, 0, q[5]);

    aSeven = transformation(0.0675, PI/2, 0, q[6]);

aOne << cos(q[0]), 0, sin(q[0]), 0,
        sin(q[0]), 0, -1 * cos(q[0]), 0,
        0, 1, 0, 0.1,
        0, 0, 0, 1;

    aTwo << cos(q[1]), 0, -1 * sin(q[1]), 0,
        sin(q[1]), 0, cos(q[1]), 0,
        0, -1, 0, 0.2,
        0, 0, 0, 1;

    aThree << cos(q[2]), 0, -1 * sin(q[2]), 0,
        sin(q[2]), 0, cos(q[2]), 0,
        0, -1, 0, 0.2,
        0, 0, 0, 1;

    aFour << cos(q[3]), 0, sin(q[3]), 0,
        sin(q[3]), 0, -1 * cos(q[3]), 0,
        0, 1, 0, 0.2,
        0, 0, 0, 1;

    aFive << cos(q[4]), 0, sin(q[4]), 0,
        sin(q[4]), 0, -1 * cos(q[4]), 0,
        0, 1, 0, 0.2,
        0, 0, 0, 1;

    aSix << cos(q[5]), 0, -1 * sin(q[5]), 0,
        sin(q[5]), 0, cos(q[5]), 0,
        0, -1, 0, 0.19,
        0, 0, 0, 1;

    aSeven << cos(q[6]), -1 * sin(q[6]), 0, 0,
        sin(q[6]), cos(q[6]), 0, 0,
        0, 0, 1, 0.0675,
        0, 0, 0, 1;

        VectorXd getJointAngles(MatrixXd& ys, MatrixXd& yds, VectorXd& startJoint, VectorXd& ts, ros::Publisher& trajPub, ros::Rate loop_rate) {
	MatrixXd angles(ts.size(), 7), angleVelocities(ts.size(), 7), J(3,7), Jp(7,3), Jt(7,3), J0(3,7);	//all 7 joint angles for all times
	VectorXd q(7), term1(7), yds_block(3), ys_block(3), ys_block_prev(3);
	VectorXd currentAngles(7); 	//to be published
	double timeStep = ts[1] - ts[0];

	for(int i = 0; i < 7; i++) {	//fist point is the starting point
		angles(0, i) = startJoint[i];
		angleVelocities(0,i) = 0;	//velocities start at zero
		currentAngles[i] = angles(0,i);
	}
	publishMessages(currentAngles, trajPub, loop_rate);
	for(int t = 1; t < ts.size(); t++) {	//ts.size()
		//q = angles.block(t-1,0,1,7);	//the joint angles from the previous time step
		for(int i = 0; i<7; i++) {
			q[i] = angles(t-1,i);
		}
		J = getJacobian(q);
		cout << "q " << q << endl;
		J0 = getJ0(q);
		Jp = getPseudoInverse(J);  	//pseudoinverse is very unstable 
		Jt = J.transpose();				//transpose is not very precise, but still better, so it's being used in term1

		//temp = Jp*yds.block(t,0,0,3).transpose();
		for(int i = 0; i<3; i++) {
			yds_block[i] = yds(t,i);		//velocity in the current time step 
			ys_block[i] = ys(t,i);
			ys_block_prev[i] = ys(t-1,i);
		}

		term1 = Jt*yds_block;	//inverse kinematics
		//null space constraints
		//x0d comes from vrep via subscriber (global variable)
		MatrixXd term2 = J0*(MatrixXd::Identity(7,7) - Jp*J);
		cout << J0 << endl;
		VectorXd term3 = getPseudoInverse(term2)*(x0d - J0*Jp*yds_block);
		for(int i = 0; i<7; i++) {		//just putting the values in the matrix
			angleVelocities(t,i) =  term1[i] + term3[i];		//add here the IK term
			//cout << temp[i] << endl;
		}
		//"integrate" to get angles for next time step
		for(int i = 0; i<7; i++) {
			angles(t,i) = angles(t-1,i) + 7*angleVelocities(t,i)*timeStep;
			currentAngles[i] = angles(t,i);
		}
		publishMessages(currentAngles, trajPub, loop_rate);
	}
	return currentAngles;
}
