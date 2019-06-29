// this program should publish messages to control the robot using dmp in combination with inverse kinematics
//it learns the easy task and saves the state of the dmp in a file to be loaded by dmp_ros_ctl_ik
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
//#include "/home/hrcstudents/Documents/Code/catkin_ws/src/dmp_bbo_node/cereal-master/include/cereal/cereal.hpp"
//#include <cereal/archives/binary.hpp>
#include "ros/ros.h"
#include "std_msgs/String.h"
#include "std_msgs/Float32.h"
#include "std_msgs/Bool.h"
#include <sstream>
#include <fstream> 
#include "geometry_msgs/Point.h"
#include "geometry_msgs/PoseStamped.h"
#include <vector>
#include <map>
#include <limits>
#include <cmath>
#include <math.h>
#include <cstdio>
#include "/home/hrcstudents/Documents/Code/catkin_ws/src/dmp_bbo_node/include/dmp_bbo_node/gnuplot-iostream.h"
#include <thread>
#include <chrono>
#include <time.h>
#include "/home/hrcstudents/Documents/Code/catkin_ws/src/dmp_bbo_node/include/dmp_bbo_node/keyboard.h"
#include "/usr/include/SDL/SDL.h"
#include <keyboard/Key.h>
#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/Eigenvalues>
#include "sensor_msgs/JointState.h"
#include "trajectory_msgs/JointTrajectory.h"
#include "trajectory_msgs/JointTrajectoryPoint.h"
//#include <tf2/LinearMath/Transform.h>
//#include "tf2_geometry_msgs/tf2_geometry_msgs.h"
//#include <tf/transform_datatypes.h>
#include <qualisys/Subject.h>
#include <std_msgs/Float64MultiArray.h>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

//#include "MatlabEngine.hpp"
//#include "MatlabDataArray.hpp"
using namespace std;
using namespace Eigen;
using namespace DmpBbo;

const double PI = 3.141592653589793238463;

ros::Publisher vrepPub;	//sends qualisys data to vrep
ros::Publisher predictionPub; //sends ptomp data to vrep 

//global variables, get filled by callback function from subscriber
Vector3d obstaclePosition = Vector3d::Zero(), obstacleVelocity = Vector3d::Zero();
int closestJoint = 2;			//number of joint closest to obstacle (from 1 to 7) 
bool collision = false;			//gets used in computeCosts
double a = 0.5, b = 0.03, c = 1;					//values in vector [a, b, c, 1], get multiplied by forward transformation to compute J0 (see Hoffmann paper)
Vector3d x0d = Vector3d::Zero();					//velocity of the closest point to the obstacle, needed for null space contraint
Vector3d handPosition = Vector3d::Zero(), handVelocity = Vector3d::Zero();
Vector3d robotBasePosition = Vector3d::Zero();
double humanDuration = 0, humanTime = 0;

double startOfRobotMovement; 	//contains the time at the start of the movement of the robot, is used to compute the human reaction time in getHumanTime()

double workingAreaCost = 0;		//variable for computeCosts, get filled out in main() 

//global ros messages for publisher (for impedance controller)
std_msgs::Float64MultiArray  msg_positions, msg_torque, msg_gains;

//function to publish trajectory messages to the robot, implemented below main()
void publishMessages(VectorXd& angles, VectorXd& currentAngleVelocities, ros::Publisher& trajPub);

//publishes the same messages but in a different format for the impedance controller
void publishMessagesImpedance(VectorXd& angles, ros::Publisher& pub_positions, ros::Publisher& pub_torque, ros::Publisher& pub_gains);

//integtrates a Matrix, with a specified startValue and a time steps, writes the result in result. was tested, it gives similar results to the computed values \o/
void integrate(VectorXd& prevValue, VectorXd& derivative, double timeStep, MatrixXd& result, int t);
//function takes costVariables and computes cost, implemented below
void computeCosts(MatrixXd& ys, MatrixXd& yds, MatrixXd& ydds, int nDims, VectorXd& cost, VectorXd& ts, bool collisionOccured, MatrixXd& angles, MatrixXd& angleVelocities, MatrixXd& ydReal, double robot_time);

//computes pseudo inverse for the jacobian, needed for the angularVelocities, impelemted below
MatrixXd getPseudoInverse(MatrixXd& J);

//computes the inverse kinematics with null-space constraints, takes the previous angles and the cartesian velocity and returns the angular velocity
VectorXd getCurrentVelocities(MatrixXd& ys, MatrixXd& yds, VectorXd& prevAngles, VectorXd& ts, int t);

//ckecks if the computed angles are within the kuka con
void checkConstraints(VectorXd& angles);

//function computes forward kinematics using ys which includes all joint angles, implemented below
Vector3d getEndEffectorPosition(VectorXd& q);

//function computes end effector velocity from joint velocities, uses getJacobian() function, implemented below
MatrixXd getEndEffectorVelocity(MatrixXd& ys, MatrixXd& yds, VectorXd& ts);

//function to get the Jacobian, implemented below, only used for the getEndEffectorVelocity function
MatrixXd getJacobian(VectorXd& q);

//matrix template, for the forward kinematics transformation matrices, input: DH parameters: d, alpha, a, theta
MatrixXd transformation(double d, double al, double a, double th);
//computes the forward kinematics for the getEndEffectorVelocity() function
MatrixXd forwardKinematics(VectorXd& q);

//saves trajectory and cost data to file 
void writeToFile(int subjectNr, int experimentNr, int updateNr, int goalNr, int sampleNr, VectorXd& data);

//gradient calculations for potential force, derived by hand
Vector3d getGradientCos(Vector3d& vel, Vector3d& pos, double dist) {
	Vector3d gradient;
	double den = pow(vel.norm()*dist,2);	//denumerator
	if(den <= 0.0001) {		//threshhold to not produce NaNs
		den = 0.0001;
	}
	//cout << "den " << endl << den << endl;
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
Vector3d dynamicPotentialForce(Vector3d& currentY, MatrixXd& yds, int nDims, VectorXd& ts, int t) {
	double lambda = 2, beta = 2;  					//parameters to include in policy improvement
	Vector3d force = Vector3d::Zero();
	Vector3d ypos = currentY;		//better to use here the closest point to obstacle, now it's the position of the end effector

	Vector3d vel;
	vel = yds.row(t);

	Vector3d x_obs, xd_obs; 		//position and velocity of the obstacle

	x_obs = obstaclePosition; 
	xd_obs = obstacleVelocity;								//obstacle position and velocity from vrep, change to handPosition and handVelocity when working with qualisys
	Vector3d pos = ypos - x_obs;	//relative position to the obstacle
	double dist = pos.norm();		//distance between obstacle and end effector p(x)

	if(dist == 0) {
		cout << "collision!!" << endl;
	}

	Vector3d rel_vel = vel - xd_obs;	//relative velocity
	//the following calculation was adapted from the dynamic potential field equations presented in Hoffmann's paper
	double temp = rel_vel.norm()*dist;
	double cos_theta = (double)(rel_vel.transpose()*pos)/temp;	
	Vector3d temp2 = rel_vel.cross(pos);
	double theta = -atan2(temp2.norm(), rel_vel.dot(pos));	//only from 0 to pi
	cos_theta = cos(theta);
	//force = lambda * pow(-cos_theta, beta-1) * rel_vel.norm()/dist * (beta*getGradientCos(vel, pos, dist) - cos_theta/dist*getGradientDistance(dist, pos));
	//cout << "theta " << theta << endl;
	//if(acos(cos_theta) < PI/2 && acos(cos_theta) > 0) {
	if(theta > -PI/2 && theta < 0) {
		force = lambda * pow(-cos_theta, beta-1) * rel_vel.norm()/dist * (beta*getGradientCos(vel, pos, dist) - cos_theta/dist*getGradientDistance(dist, pos));
		//cout << "force was applied. right" << endl;
	}	
	//else  if (acos(cos_theta) > -PI/2 && acos(cos_theta) < 0) {
	else  if (theta > -PI/2 && theta < 0) {
		cos_theta = - cos_theta;
		force = lambda * pow(-cos_theta, beta-1) * rel_vel.norm()/dist * (beta*getGradientCos(vel, pos, dist) - cos_theta/dist*getGradientDistance(dist, pos));
		//force = Vector3d::Zero();
		//cout << "force was applied. left" << endl;
	}
	else {
		//force = Vector3d::Zero();
	}
	return force;
}

//callback functions for all the subscribers, save data in global variables declared above
void getObstaclePosition(geometry_msgs::Point point) {
	obstaclePosition[0] = point.x;
	obstaclePosition[1] = point.y;
	obstaclePosition[2] = point.z;
	//cout << "obstacle position callback " << endl << obstaclePosition << endl; 
}

void getLinkVelocity(geometry_msgs::Point msg) {
	x0d[0] = msg.x;
	x0d[1] = msg.y;
	x0d[2] = msg.z;
}

void getClosestJoint(std_msgs::Float32 msg) {
	closestJoint = (int)msg.data;
}

void getDistanceToJoint(geometry_msgs::Point point) {
	a = point.x;
	b = point.y;
	c = point.z;
}

void getObstacleVelocity(geometry_msgs::Point msg) {
	obstacleVelocity[0] = msg.x;
	obstacleVelocity[1] = msg.y;
	obstacleVelocity[2] = msg.z;
}

void setCollisionStatus(std_msgs::Bool status) {
	collision = status.data;
	//cout << "collision: " << collision << endl;
}

double handSubTime = 0;		//time between the position messages to compute the velocity
void getHandPosition(qualisys::Subject msg) {
	handSubTime = ros::Time::now().toSec() - handSubTime;		//the time since the last message
	handVelocity[0] = (msg.position.x - handPosition[0])/handSubTime;	//the difference between the last and the current position
	handVelocity[1] = (msg.position.y - handPosition[1])/handSubTime;
	handVelocity[2] = (msg.position.z - handPosition[2])/handSubTime;

	handPosition[0] = msg.position.x;
	handPosition[1] = msg.position.y;
	handPosition[2] = msg.position.z;

	geometry_msgs::Point pos;
	pos = msg.position;
	//cout << "publishing position" << endl;
	vrepPub.publish(pos);		//publish to vrep to do distance etc. computation
	ros::spinOnce();	
}

void getPtomp(const sensor_msgs::JointState::ConstPtr& msg)
{
	MatrixXd ptompData;
	cout << "size ptomp " << msg->position.size() << endl;
    string predictedGoal = msg->name[0];
    for (int i = 0; i < 51; i++) {
        ptompData(i, 0) = msg->position[i];
        ptompData(i, 1) = msg->velocity[i];
        ptompData(i, 2) = msg->effort[i];
    }
    cout << "ptomp received " << endl;
    geometry_msgs::Point futurePos; 
    futurePos.x = ptompData(15,0);	//one point in the future (chosen arbitrarily)
    futurePos.y = ptompData(15,1);
    futurePos.z = ptompData(15,2);
    predictionPub.publish(futurePos);
    ros::spinOnce();
}

void callRobotFrame(qualisys::Subject msg) {
	robotBasePosition[0] = msg.position.x;
	robotBasePosition[1] = msg.position.y;
	robotBasePosition[2] = msg.position.z;
}

void getHumanTime(std_msgs::Float32 msg) {
	if(msg.data < 4) {
		humanDuration = msg.data;
	}
	double end = ros::Time::now().toSec();
	if(end - startOfRobotMovement < 5 && end - startOfRobotMovement > 0.01) {
		humanTime = end - startOfRobotMovement;
	}
	cout << "received human time" << endl;
}

//due to lack of straight forward method to create 3d arrays, this function puts different matrices beside each other in a big matrix
void addToBigMatrix(MatrixXd& newMatrix, MatrixXd& bigMatrix, int ithMatrix) {
	for(int i = 0; i < newMatrix.rows(); i++) {
		for(int j = 0; j < newMatrix.cols(); j++) {
			bigMatrix(i, j+ithMatrix*newMatrix.cols()) = newMatrix(i,j);
		}
	}
}

//global variable, stores all the angle velocities for each trajectory, angles stores all angles for each trajectory
MatrixXd angleVelocities, angles, angleAccelerations;

//////////////////////////////////////////////////////////////DMP Solver Class/////////////////////////////////////////////////////////////////////////////////////////////
class DmpSolver 
{
private:
	DmpSolver();
	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version)
	{
		ar & _goalSys;
		ar & _phaseSys;
		ar & _gatingSys;
		ar & _trajectory;
		ar & _metaParam;
		ar & _faRbfn;
		ar & _dmp;
		ar & _taskSolver;
		ar & _distribution;
		ar & _updater;
	    ar & _tau;
	    ar & _start;
	    ar & _goal;
	    ar & _dt;
	    ar & _meanInit;
	    ar & _covarSize;
	    ar & _alphaGoal;
	    ar & _nBasisFunc;
	    ar & _alphaSpringDamper;
	    ar & _integrate_dmp_beyond_tau_factor;
	    ar & _ts;
	    ar & _nDims;
	    ar & _eliteness;
	    ar & _covarDecayFactor;
	    ar & _meanCost;

	}
public: 
	DmpSolver(double tau, VectorXd& start, VectorXd& goal, double dt, double covarSize, int nBasisFunc, double alphaGoal, double alphaSpringDamper, double eliteness, double covarDecayFactor, double integrate_dmp_beyond_tau_factor);	//constructor
	~DmpSolver(); 	//destructor
	void generateSamples(int nSamplesPerUpdate, MatrixXd& samples);
	void performRollouts(MatrixXd& samples, MatrixXd& costVariables);
	void updateDistribution(MatrixXd& samples, VectorXd& costPerSample);

	DynamicalSystem *_goalSys;
	DynamicalSystem *_phaseSys;
	DynamicalSystem *_gatingSys;
	Trajectory _trajectory;
	MetaParametersRBFN *_metaParam;
	FunctionApproximatorRBFN *_faRbfn;
	Dmp *_dmp;
	TaskSolverDmp *_taskSolver;
	DistributionGaussian *_distribution;
	Updater *_updater;

	double _tau;
	VectorXd _start;
	VectorXd _goal;
	double _dt;
	VectorXd _meanInit;
	double _covarSize = 300;	//the bigger the variance, the bigger the sample space
	double _alphaGoal = 20;		//decay constant
	int _nBasisFunc = 20;
	double _alphaSpringDamper = 20;
	double _integrate_dmp_beyond_tau_factor = 1.2;
	VectorXd _ts;	//vector of time steps
	int _nDims = 3;
	double _eliteness = 10;				//values from old code
    double _covarDecayFactor = 0.9;		//sample space gets smaller with time
    double _meanCost = 0;
};

DmpSolver::DmpSolver() 
{

}

DmpSolver::DmpSolver(double tau, VectorXd& start, VectorXd& goal, double dt, double covarSize=300, int nBasisFunc=20, double alphaGoal=20, double alphaSpringDamper=20, double eliteness=10, double covarDecayFactor=0.9, double integrate_dmp_beyond_tau_factor=1.2) 
	:_tau(tau), _start(start), _goal(goal), _dt(dt), _covarSize(covarSize), _alphaGoal(alphaGoal), _nBasisFunc(nBasisFunc), _alphaSpringDamper(alphaSpringDamper), 
	_integrate_dmp_beyond_tau_factor(integrate_dmp_beyond_tau_factor), _nDims(goal.size())
{
	VectorXd one  = VectorXd::Ones(1);
 	VectorXd zero = VectorXd::Zero(1);	
	_goalSys   = new ExponentialSystem(tau, start, goal, _alphaGoal);
	_phaseSys  = new TimeSystem(tau);
	_gatingSys = new ExponentialSystem(tau, one, zero, 5);				//starts at one, ends at 0, decay const = 5, ensures convergence, 1D

	int timeSteps = ceil(tau*integrate_dmp_beyond_tau_factor/dt)+1;	//number of time steps 
	_ts = VectorXd::LinSpaced(timeSteps, 0.0, _tau);			//vector with all time steps
	_trajectory = Trajectory::generateMinJerkTrajectory(_ts, _start, _goal);

	_metaParam = new MetaParametersRBFN(1, _nBasisFunc);
	_faRbfn = new FunctionApproximatorRBFN(_metaParam);

	vector<FunctionApproximator*> _funcAppr;
	for (int i = 0; i < _nDims; i++) {
        _funcAppr.push_back(_faRbfn->clone());
	}

	_dmp = new Dmp(_nDims, _funcAppr, _alphaSpringDamper, _goalSys, _phaseSys, _gatingSys);
	_dmp->train(_trajectory);

	set<string> parameters_to_optimize;
    parameters_to_optimize.insert("weights");		//we optimize the weights of the RBFN
    _taskSolver = new TaskSolverDmp(_dmp, parameters_to_optimize, _dt, _integrate_dmp_beyond_tau_factor);

    _dmp->getParameterVectorSelected(_meanInit);	//-> centers around the weights
    MatrixXd covarInit = _covarSize * MatrixXd::Identity(_meanInit.size(), _meanInit.size());
    _distribution = new DistributionGaussian(_meanInit, covarInit);		//distribution around the parameters to sample

    string weightingMethod("PI-BB");	//BB: reward-weighted averaging
    _updater = new UpdaterCovarDecay(_eliteness, _covarDecayFactor, weightingMethod);
}

void DmpSolver::generateSamples(int nSamplesPerUpdate, MatrixXd& samples){
	_distribution->generateSamples(nSamplesPerUpdate, samples);
}

void DmpSolver::performRollouts(MatrixXd& samples, MatrixXd& costVariables) {
	_taskSolver->performRollouts(samples, costVariables);			//costVariables now includes y_out, yd_out, ydd_out, ts, forcing terms for each sample
}

void DmpSolver::updateDistribution(MatrixXd& samples, VectorXd& costPerSample) {
	_updater->updateDistribution(*_distribution, samples, costPerSample, *_distribution);
}

DmpSolver::~DmpSolver() {
	delete _goalSys;
	delete _phaseSys;
	delete _gatingSys;
	delete _metaParam;
	//delete _funcAppr;
	delete _faRbfn;
	delete _dmp;
	delete _taskSolver;
	delete _distribution;
	delete _updater;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//contains the update loop, is the MOST IMPORTANT function, the core of this program
int executeTrajectories(DmpSolver* dmpSolver[], MatrixXd& goal, int nGoals, VectorXd& startJoint, ros::Publisher& trajPub, ros::Rate& loop_rate, int nUpdates, int nSamplesPerUpdate, int subjectNr, int experimentNr);

void initRosMsgs() {
	msg_positions.data.resize(7);
	msg_gains.data.resize(14);
	msg_torque.data.resize(7);

	for(int i=0; i<7; i++)
		{
			msg_gains.data[i] =  700;
			msg_gains.data[i+7] = 0.6;
		}

	for(int i=0; i<7; i++)
		msg_torque.data[i] = 0;
}
////////////////////////////////////////////////////////////////////////////////////main///////////////////////////////////////////////////////////////////////////////////
ofstream distanceFile;
ros::Publisher endeffPub;
ros::Publisher forcePub;

ros::Publisher pub_positions;
ros::Publisher pub_gains;
ros::Publisher pub_torque;

int main(int argc, char *argv[])
{
	//define node
	ros::init(argc, argv, "dmp_ik");
	ros::NodeHandle node;
	
	//test
	double t = ros::Time::now().toSec();
	string tes = str(boost::format("testd f %1%") % 2);
	cout << tes.c_str() << endl;

	//define publisher, Robot Publisher
	string trajTopic("/standing2/joint_trajectory_controller/command");
	ros::Publisher trajPub = node.advertise<trajectory_msgs::JointTrajectory>(trajTopic.c_str(), 800);		
	ros::Rate loop_rate(55);
	pub_positions = node.advertise<std_msgs::Float64MultiArray>("/standing2/joint_impedance_controller/position", 100);
	pub_gains = node.advertise<std_msgs::Float64MultiArray>("/standing2/joint_impedance_controller/gains", 100);
	pub_torque = node.advertise<std_msgs::Float64MultiArray>("/standing2/joint_impedance_controller/torque", 100);

	initRosMsgs();

	//publisher to publish cartesian position of end effector (for debugging) 
	endeffPub = node.advertise<geometry_msgs::Point>("/positionEndeffector", 1);
	forcePub  = node.advertise<geometry_msgs::Point>("/force", 1);
	//publisher to publish the qualisys msgs to vrep
	vrepPub = node.advertise<geometry_msgs::Point>("/wristForVrep", 100);
	predictionPub = node.advertise<geometry_msgs::Point>("predictionForVrep", 100);
	//define subscriber to get obstacle position and closestPointToObstacle position (we need closest joint and distance to closest joint) from vrep
	ros::Subscriber obsSub   = node.subscribe("/position/obstacle", 1, getObstaclePosition);	// --> this subscriber is for working with vrep
	ros::Subscriber jointPosSub = node.subscribe("/x0d", 1, getLinkVelocity);
	ros::Subscriber jointSub = node.subscribe("/closestJoint", 1, getClosestJoint);
	ros::Subscriber distSub  = node.subscribe("/distanceToJoint", 1, getDistanceToJoint);
	ros::Subscriber obsVelSub = node.subscribe("/obstacleVelocity", 1, getObstacleVelocity);
	//define subscriber for collision status from vrep
	ros::Subscriber collSub = node.subscribe("/collision_status", 1, setCollisionStatus);		//not used
	//subscriber for qual0.02 (gets obstacle/hand position) 
	ros::Subscriber handSub = node.subscribe("/qualisys/wrist_right", 100, getHandPosition);		// subscriber to the camera system
	handSubTime = ros::Time::now().toSec();
	//subscriber to get the time the human needs to reach the goal (from the prediction code)
	ros::Subscriber humanTimeSub = node.subscribe<std_msgs::Float32>("/motion_duration", 10, getHumanTime);
	//ros::Subscriber getPtompSub = node.subscribe<sensor_msgs::JointState>("/JS_predict_wrist", 10, getPtomp);
	//subscriber to get the position of the base of the robot from the camera system
	ros::Subscriber robotFrameSub = node.subscribe("/qualisys/standing2", 100, callRobotFrame);		//not used

////////////////////////////////////////////////////////////define start and goal position(s)///////////////////////////////////////////////////////////////
	//define goal and start
	//VectorXd startJoint(7); startJoint << 80.44 * PI / 180, 17.22 * PI / 180, 0* PI / 180,-70.36 * PI / 180, 0 * PI / 180, 0, 0;	//initial state in joint space
	//VectorXd startJoint(7); startJoint << -40.74 * PI / 180, -28 * PI / 180, -156.35* PI / 180, -101.99 * PI / 180, -94.31 * PI / 180, 0, 0;
	VectorXd startJoint(7); startJoint << -153.49 * PI / 180, -37.16 * PI / 180, -149.82* PI / 180, -91.73 * PI / 180, -86.12 * PI / 180, 0, 0;
	//VectorXd startJoint(7); startJoint << 0,90*PI/180,0,-90*PI/180,0,0,0;
	VectorXd start(3); start = getEndEffectorPosition(startJoint);													//initial state in cartesian space
	cout << "start " << endl << start << endl;

	int nGoals = 4; //3;													//the number of goals
	//MatrixXd goal(nGoals,3); goal << -0.45, -0.9, 0.84,	//goal 1	make this longer if you want more goals (add the positions)
	//								 -0.75, -0.9, 0.84,	//goal 2
	//								 -1.05, -0.9, 0.84;	//goal 3
	MatrixXd goal(nGoals,3); goal << -1.104, -0.8759, 0.95,	//goal 1
									 -0.8982, -0.9798, 0.95,	//goal 2
									 -0.7026,-1.0465, 0.95,	//goal 3
									 -0.5306, -0.8762, 0.95;	//goal 4

	///////////////////////////////////////////////////define the DMP///////////////////////////////////////////////////////////////////////////////////////////////
	double tau = 2.5;		//time constant
	double dt = 0.02;		//integration step duration
	double covarSize = 300; //the bigger the variance, the bigger the sample space
	int nBasisFunc = 20;	//n gaussians per dmp

	DmpSolver *dmpSolver[nGoals]; 
	for(int i = 0; i < nGoals; i++) {
		VectorXd tempGoal = goal.block(i,0,1,3).transpose();
		dmpSolver[i] = new DmpSolver(tau, start, tempGoal, dt, covarSize);	//initialize dmps for all goals
	}
	
	int subjectNr, experimentNr;
	cout << "Which subject is doing the experiment?" << endl;
	cin >> subjectNr;
	cout << "Experiment number?" << endl;
	cin >> experimentNr;
	//////////////////////////////////////////////////////habituation phase///////////////////////////////////////////////////////////////////////////////////////////
	//rollout one trajectory per goal for the habituation phase
	int nUpdates = 1;		
    int nSamplesPerUpdate = 1;
    executeTrajectories(dmpSolver, goal, nGoals, startJoint, trajPub, loop_rate, nUpdates, nSamplesPerUpdate, subjectNr, experimentNr);	
    /////////////////////////////////////////////////////run the policy improvement////////////////////////////////////////////////////////////////////////////////
	nUpdates = 10;		
    nSamplesPerUpdate = 4;

    cout << "Press enter to start the experiment." << endl;
    cin.ignore();

	dmpSolver[0]->_meanCost = executeTrajectories(dmpSolver, goal, nGoals, startJoint, trajPub, loop_rate, nUpdates, nSamplesPerUpdate, subjectNr, experimentNr);	//do the magic

	/////////////////////////////////////////////////////////archive/////////////////////////////////////////////////////////////////////////////////////////////////
	for(int i = 0; i<nGoals; i++) {
		string fileName = str(boost::format("Results/trainedDmpPolicy_G%1%_S%2%_E%3%.txt") % i % subjectNr % experimentNr);	
		ofstream ofs(fileName);
		boost::archive::text_oarchive oa(ofs);
		oa << dmpSolver[i];

		cout << "archive for goal successful " << i << endl;
	}
/*	{
	ofstream ofs("trainedDmpPolicy1.2");
	boost::archive::text_oarchive oa(ofs);
	oa << dmpSolver[0];

	cout << "archive for first goal successful" << endl;
	}
	{
	ofstream ofs("trainedDmpPolicy2.2");
	boost::archive::text_oarchive oa(ofs);
	oa << dmpSolver[1];

	cout << "archive for second goal successful" << endl;
	}*/
	//destroy objects
    //delete [] dmpSolver;
	
	return 0;
}

////////////////////////////////////////////////////////////////////////////////publish messages//////////////////////////////////////////////////////////////////////////////////////////////////
void publishMessages(VectorXd& angles, VectorXd& currentAngleVelocities, ros::Publisher& trajPub) {	
	
	trajectory_msgs::JointTrajectory msg;

    msg.joint_names.clear();
    msg.joint_names.push_back("standing2_a1_joint");
    msg.joint_names.push_back("standing2_a2_joint");
    msg.joint_names.push_back("standing2_a3_joint");
    msg.joint_names.push_back("standing2_a4_joint");
    msg.joint_names.push_back("standing2_a5_joint");
    msg.joint_names.push_back("standing2_a6_joint");
    msg.joint_names.push_back("standing2_e1_joint");
    msg.points.resize(1);						//one point with 7 positions, 7 velocities and 7 accelerations for 7 joints
    msg.points[0].positions.resize(7); 
    msg.points[0].velocities.resize(7);
    msg.points[0].accelerations.resize(7);
    msg.points[0].time_from_start = ros::Duration(0.015*2);		//a bit longer than dt, multiplied with 4, because of t%4 above

    msg.points[0].positions[0] = angles[0];
    msg.points[0].positions[1] = angles[1]; //+PI/2;
    msg.points[0].positions[6] = angles[2];		//the third joint in the kuka is last (configuration of the robot)
    msg.points[0].positions[2] = angles[3];
    msg.points[0].positions[3] = angles[4];
    msg.points[0].positions[4] = angles[5];
    msg.points[0].positions[5] = angles[6];

	msg.points[0].velocities[0] = currentAngleVelocities[0];	//should be negative in the reverse roll-out..
	msg.points[0].velocities[1] = currentAngleVelocities[1];
	msg.points[0].velocities[6] = currentAngleVelocities[2];
	msg.points[0].velocities[2] = currentAngleVelocities[3];
	msg.points[0].velocities[3] = currentAngleVelocities[4];
	msg.points[0].velocities[4] = currentAngleVelocities[5];
	msg.points[0].velocities[5] = currentAngleVelocities[6];
	trajPub.publish(msg);
}


void publishMessagesImpedance(VectorXd& angles, ros::Publisher& pub_positions, ros::Publisher& pub_torque, ros::Publisher& pub_gains){
	msg_positions.data[0] = angles[0];
    msg_positions.data[1] = angles[1]; //+PI/2;
    msg_positions.data[2] = angles[2];		//the third joint in the kuka is last (configuration of the robot)
    msg_positions.data[3] = angles[3];
    msg_positions.data[4] = angles[4];
    msg_positions.data[5] = angles[5];
    msg_positions.data[6] = angles[6];

    pub_gains.publish(msg_gains);
    pub_torque.publish(msg_torque);
    pub_positions.publish(msg_positions);
}
///////////////////////////////////////////////////////////////////////////cost function/////////////////////////////////////////////////////////////////////////////////////////////////////////
void computeCosts(MatrixXd& ys, MatrixXd& yds, MatrixXd& ydds, int nDims, VectorXd& cost, VectorXd& ts, bool collisionOccured, MatrixXd& angles, MatrixXd& angleVelocities, MatrixXd& ydReal, double robot_time) {
	//maybe add another cost that the dmp has to stay within the limits of the robot (within a cartesian radius)
	double wHumanT = 10, wRobot = 1, wAccuracy = 10, wJoint = 1, wEndeffector = 1, wCollision = 1000, wTrajectory = 2, wHumanD = 5;		//weights to make all variables in the same range
	double human_time = 1, accuracy = 0, joint_jerk = 1, endeffector_jerk = 1, trajectory = 0, human_duration = 0;
	double deltaT = ts[1] - ts[0];

	for(int i = 0; i < cost.size(); i++) {
		cost[i] = 0;
	}

	human_time = humanTime;
	human_duration = humanDuration*2; //normalized duration of human movement
	cout << "Was the prediction correct? (1 if correct, 0 if incorrect) " << endl;
	cin >> accuracy; 
	accuracy = !accuracy;	//if it was correct -> no cost, if it was incorrect -> cost = 1

	//difference between the direct output of the dmp and the deformed trajectory, over time the dmp should get closer to the deformed trajectory
	VectorXd posEndEffDmp(3), posEndeffReal(3);
	for(int t = 0; t < ts.size(); t++) {
		posEndEffDmp[0] = ys(t,0);
		posEndEffDmp[1] = ys(t,1);
		posEndEffDmp[2] = ys(t,2);

		VectorXd currentAngles = angles.block(t,0,1,7).transpose();
		posEndeffReal = getEndEffectorPosition(currentAngles);
		VectorXd difference = posEndEffDmp - posEndeffReal;
		trajectory += difference.norm();

	}
	//cout << "trajectory cost " << trajectory << endl;

	//calculate end-effector jerk with obstacle avoidance
	MatrixXd accelerations = MatrixXd::Zero(ts.size(),3);
	for(int t = 0; t < ts.size()-1; t++) {
		for(int j = 0; j < 3; j++) {
			accelerations(t,j) = (ydReal(t+1,j)-ydReal(t,j))/deltaT;		//derive acceleration
		}
	}
	for(int i = 0; i < ts.size()-1; i++) {
		for(int j = 0; j < 3; j++) {
			endeffector_jerk += abs(accelerations(i+1,j) - accelerations(i,j))/deltaT;		//derive acceleration
		}
	}
	//cout << "endeffector cost " << endeffector_jerk << endl;

	//calculate joint jerk
	MatrixXd jointAccelerations = MatrixXd::Zero(ts.size(),7);
	for(int t = 0; t < ts.size()-1; t++) {
		for(int j = 0; j < 7; j++) {
			jointAccelerations(t,j) = (angleVelocities(t+1,j)-angleVelocities(t,j))/deltaT;		//derive acceleration
		}
	}
	for(int i = 0; i < ts.size()-1; i++) {
		for(int j = 0; j < 7; j++) {
			joint_jerk += abs(jointAccelerations(i+1,j) - jointAccelerations(i,j))/deltaT;		//derive acceleration
		}
	}
	//cout << "joint cost " << joint_jerk << endl;
	//cout << "collision: " << collision << endl;

	cost[1] = human_time*wHumanT;	
	cout << "human time " << human_time << endl;	
	cost[2] = robot_time*wRobot;
	cost[3] = accuracy*wAccuracy;
	//cost[4] = joint_jerk*wJoint;
	cost[4] = (joint_jerk-5500)/(1000000-550000)*wJoint;
	cout << "joint cost " << cost[4] << endl;
	//cost[5] = endeffector_jerk*wEndeffector;
	cost[5] = (endeffector_jerk-2000)/(210000-60000)*wEndeffector;
	cout << "endeffector cost " << cost[5] << endl;
	//cost[6] = wCollision*collisionOccured;
	cost[6] = wTrajectory*(trajectory-1)/5;
	cout << "trajectory cost " << cost[6] << endl;
	//cost[7] = wHumanD*human_duration;
	cost[7] = 0;
	cost[0] = cost.sum();

}
//////////////////////////////////////////////////////////////////////////////////////////update loop//////////////////////////////////////////////////////////////////////
int executeTrajectories(DmpSolver* dmpSolver[], MatrixXd& goal, int nGoals, VectorXd& startJoint, ros::Publisher& trajPub, ros::Rate& loop_rate, int nUpdates, int nSamplesPerUpdate, int subjectNr, int experimentNr) {
	ofstream myfile; 		//for saving the cost results in a file for plotting 
	//define some values
    VectorXd ts = dmpSolver[0]->_ts;		//vector of time steps for the trajectory
    int nDims = dmpSolver[0]->_nDims;
    int nBasisFunc = dmpSolver[0]->_nBasisFunc;
    double timeStep = ts[1] - ts[0];
    double tau = dmpSolver[0]->_tau;

    angleVelocities = MatrixXd::Zero(ts.size(), 7);				//initialization
	angles = MatrixXd::Zero(ts.size(), 7);
	angleAccelerations = MatrixXd::Zero(ts.size(),7);
    
    VectorXd costs(8); //has the detailed costs, gets filled by compute costs, costs[0] is the totat cost

    MatrixXd samples(nSamplesPerUpdate, nDims*nBasisFunc), costVariables(nSamplesPerUpdate, (4*nDims+1)*ts.size());
    MatrixXd allSamples(samples.rows(), samples.cols()*nGoals), allCostVariables(costVariables.rows(), costVariables.cols()*nGoals);

    int meanCost = 0;

	for(int iUpdate = 0; iUpdate < nUpdates; iUpdate++) {
    	cout << "Update Nr: " << iUpdate << endl;

    	for(int i = 0; i < nGoals; i++) {
	    	dmpSolver[i]->generateSamples(nSamplesPerUpdate, samples);		//you generate multiple samples
	    	dmpSolver[i]->performRollouts(samples, costVariables);			//costVariables now includes y_out, yd_out, ydd_out, ts, forcing terms for each sample
	    	
	    	addToBigMatrix(samples, allSamples, i);							//save values in a big matrix, needed for the update
	    	addToBigMatrix(costVariables, allCostVariables, i);
	    }

		MatrixXd y, yd, ydd;								//the trajectory values that will be published

    	int nCostVariables = 4*nDims+1; 			//y, yd, ydd, ts, forcing terms
    	MatrixXd row(1, costVariables.cols());		//temporary variable
    	MatrixXd rollout;							//to store the rollout data for each sample
    	MatrixXd costPerSample(nSamplesPerUpdate, nGoals);

    	//sample loop
    	VectorXd s = VectorXd::Zero(nGoals);	//vector of counters for all samples for all goals
    	while(s.sum() < nSamplesPerUpdate*nGoals) {	//while (all samples have not been executed) any element of s is below the sample amount	
			
			MatrixXd shortCostVariables;
			Vector3d currentGoal;
    		int a = rand() % nGoals;	//random number between 0 and nGoals-1
    		if(s[a] < nSamplesPerUpdate) {	//make sure we execute all different samples for all goals
    			cout << "targeted goal: " << a+1 << endl;
	    		shortCostVariables = allCostVariables.block(0, costVariables.cols()*a, costVariables.rows(), costVariables.cols());	//chose a random block of the costVariables to execute
				currentGoal = goal.block(a,0,1,3).transpose();
    		}
    		else {
    			continue;		//go to the end of the loop and try another a 
    		}
    		cout << "Press enter to roll out trajectory." << endl;
    		cin.ignore();

    		bool collisionOccured = 0;

    		row = shortCostVariables.row(s[a]);
			rollout = (Map<MatrixXd>(row.data(), nCostVariables, ts.size())).transpose();
    		y = rollout.block(0, 0, ts.size(), nDims);
    		yd = rollout.block(0, nDims, ts.size(), nDims);
    		ydd = rollout.block(0, 2*nDims, ts.size(), nDims);

    		workingAreaCost = 0;
    		//time loop/execution loop
    		VectorXd currentAngles(7), currentAngleVelocities = VectorXd::Zero(7), currentAcceleration = VectorXd::Zero(7);
    		MatrixXd correctiveVelocity = MatrixXd::Zero(ts.size(),7);	//get integrated from the potential force
    		VectorXd currentCorrectiveVelocity = VectorXd::Zero(7);
    		MatrixXd realYd = MatrixXd::Zero(ts.size(),3);		//the cartesian velocities of the deformed trajectory (for cost computation)

    		double duration = 0; //duration for one trajectory

    		currentAngles = startJoint;
			angles.block(0,0,1,7) = currentAngles.transpose();	//save it in the big matrix

			int samplingRate = 3; 	//every 3rd point gets published -> discritization

    		for(int t = 0; t < ts.size(); t++) {

    			if(t == 0) {
    				startOfRobotMovement = ros::Time::now().toSec();
    			}
    			Vector3d currentY;
	    		currentY = y.row(t);
	    		currentY = getEndEffectorPosition(currentAngles);

	    		//publish end effector cartesian position (for testing)
				geometry_msgs::Point pos;
				pos.x = y(t,0);
				pos.y = y(t,1);
				pos.z = y(t,2);
				endeffPub.publish(pos);
				ros::spinOnce();

		    	Vector3d force = dynamicPotentialForce(currentY, yd, nDims, ts, t)/tau;		//force for obstacle avoidance	
		    	geometry_msgs::Point forcemsg;
		    	forcemsg.x = force[0];
		    	forcemsg.y = force[1];
		    	forcemsg.z = force[2];
		    	forcePub.publish(forcemsg);		//for visualization only 
		    	ros::spinOnce();				//need if we have a subscribtion in the program

				MatrixXd J = getJacobian(currentAngles);
				VectorXd currentTorque = J.transpose()*force;	//transform force into joint torque
				integrate(currentCorrectiveVelocity, currentTorque, timeStep, correctiveVelocity, t);	//integrate to add directly to velocity
				currentCorrectiveVelocity = correctiveVelocity.block(t,0,1,7).transpose();


				VectorXd posEndEff = getEndEffectorPosition(currentAngles);
				//currentCorrectiveVelocity = currentCorrectiveVelocity*(posEndEff-currentGoal).norm()*1.7;

				
				if((posEndEff-currentGoal).norm() < (posEndEff-obstaclePosition).norm()/4) {		//if the endeffector is closer to the robot than to the obstacle, it should not do avoiding maneuvers
					//currentCorrectiveVelocity = VectorXd::Zero(7);
					//cout << "corrective velocity " << endl << currentCorrectiveVelocity << endl;
				}


				currentAngleVelocities = getCurrentVelocities(y, yd, currentAngles, ts, t) + currentCorrectiveVelocity;	//save the new velocities (without modification) in the variable angleVelocities

				angleVelocities.block(t,0,1,7) = currentAngleVelocities.transpose();
				realYd.block(t,0,1,3).transpose() = J*currentAngleVelocities;

				integrate(currentAngles, currentAngleVelocities, timeStep, angles, t);
				currentAngles = angles.block(t,0,1,7).transpose();

				posEndEff = getEndEffectorPosition(currentAngles);
				if(posEndEff[2] < 0.86 && t>0) {		//stop the end effector from colliding with the table, no guarantee for the joints!
					currentAngles = angles.block(t-1,0,1,7).transpose();
					angles.block(t,0,1,7).transpose() = currentAngles;
				}

				checkConstraints(currentAngles);	//check if the angles fit the constraints before publishing

				if(t%samplingRate == 0) {
					//publishMessages(currentAngles, currentAngleVelocities, trajPub);
					publishMessagesImpedance(currentAngles, pub_positions, pub_torque, pub_gains);
					ros::spinOnce();				//need if we have a subscribtion in the program
					loop_rate.sleep();
				}	
				if(collision) {
					//collisionOccured = 1;
				}
				if(t == ts.size()-1) {
					duration = ros::Time::now().toSec() - startOfRobotMovement;
					cout << "duration for this robot trajectory: " << duration << endl;
				}
			}
			computeCosts(y, yd, ydd, nDims, costs, ts, collisionOccured, angles, angleVelocities, realYd, duration);	
    		costPerSample(s[a],a) = costs[0];

    		//int subjectNr, int experimentNr, int updateNr, int goalNr, int sampleNr, VectorXd& data
    		writeToFile(subjectNr, experimentNr, iUpdate, a+1, s[a]+1, costs);

    		cin.ignore(); 	//pause before going back

    		//bring the robot back to the start position (publish messages in reversed order)
			for(int tReverse = ts.size(); tReverse > 0; tReverse--) {
				VectorXd ang(7), vel(7);
				for(int i = 0; i < 7; i++) {
					ang[i] = angles(tReverse-1, i);
					vel[i] = -angleVelocities(tReverse-1, i);	//negative velocity because of negative direction
				}
				//cout << "reverse " << endl;
				if(tReverse%samplingRate == 0) {
					//publishMessages(ang, vel, trajPub);
					publishMessagesImpedance(ang, pub_positions, pub_torque, pub_gains);
					ros::spinOnce();				//need if we have a subscribtion in the program
					loop_rate.sleep();
				}	
			}
			s[a]++; 	//keep track of which samples we executed
    	}
    	for(int i = 0; i < nGoals; i++) {
    		MatrixXd shortSamples = allSamples.block(0, samples.cols()*i, samples.rows(), samples.cols());	
    		VectorXd shortCostPerSample = (VectorXd)costPerSample.block(0,i, costPerSample.rows(), 1);
    		dmpSolver[i]->updateDistribution(shortSamples, shortCostPerSample);
		}

    	for(int i = 0; i < costPerSample.rows(); i++) {		//for testing
			cout << "costPerSample: " << costPerSample(i,0) << ", " << flush;		//index has to be corrected
			meanCost += costPerSample(i,0);
    	}
    	meanCost /= costPerSample.rows();
    	cout << "\n" << endl;
	}
	cout << "You are done!" << endl;
	return meanCost;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void writeToFile(int subjectNr, int experimentNr, int updateNr, int goalNr, int sampleNr, VectorXd& data) {
	string fileName = str(boost::format("Results/S%1%_E%2%_U%3%_G%4%_Sa%5%.txt") % subjectNr % experimentNr % updateNr % goalNr % sampleNr);
	ofstream file;
	file.open(fileName);
	file << data;
	file.close();
}


MatrixXd transformation(double d, double al, double a, double th){
	MatrixXd matrix(4,4);
	matrix << cos(th), -sin(th)*cos(al),  sin(th)*sin(al), a*cos(th),
        	  sin(th),  cos(th)*cos(al), -cos(th)*sin(al), a*sin(th),
       		   0,           sin(al),           cos(al),        d,
        	   0,             0,                  0,            1;
	return matrix;
}

MatrixXd forwardKinematics(VectorXd& q) {
	MatrixXd aOne(4, 4), aTwo(4, 4), aThree(4, 4), aFour(4, 4), aFive(4, 4), aSix(4, 4), aSeven(4, 4), transformMat(4, 4);
	//the DH parameters have been aquired by inspection according to the DH norm
	aOne = transformation(0,    PI/2, 0, q[0]);

    aTwo = transformation(0,   -PI/2, 0, q[1]);

    aThree = transformation(0.4,-PI/2, 0, q[2]);

    aFour = transformation(0,   PI/2, 0, q[3]);

    aFive = transformation(0.39,PI/2, 0, q[4]);

    aSix = transformation(0,   -PI/2, 0, q[5]);

    aSeven = transformation(0, 0, 0, q[6]);

   	transformMat = aOne * aTwo * aThree * aFour * aFive * aSix * aSeven;
   	return transformMat;
}

Vector3d getEndEffectorPosition(VectorXd& q){
	
	MatrixXd transformMat(4,4);
	Vector3d position;
	transformMat = forwardKinematics(q);
	VectorXd temp(4); 
	VectorXd offsetEndEff(4); offsetEndEff << 0, 0, 0.078, 1;
	temp = transformMat*offsetEndEff;		//not in world frame -> has to be multiplied with matrix

	position[0] = temp[0];
	position[1] = temp[1];
	position[2] = temp[2];
	
	Vector3d offsetBase; offsetBase << 0, 0, 0.31;
	Vector3d offsetTable; offsetTable << -0.8, -0.275, 0.75; 	//this is with respect to the world frame
	return offsetTable + offsetBase + position;
	//return robotBasePosition + position;
}

MatrixXd getJacobian(VectorXd& q) {
	MatrixXd jacobian(3,7); //3 cartesian dimensions, 7 joints
	double q1 = q[0], q2 = q[1], q3 = q[2], q4 = q[3], q5 = q[4], q6 = q[5], q7 = q[6];
	//the following jacobian was computed symbolically with MATLAB with the forward kinematics in the getEndEffectorPosition function
	jacobian(0,0) = (2*sin(q1)*sin(q2))/5 - (39*cos(q6)*(sin(q4)*(cos(q1)*sin(q3) + cos(q2)*cos(q3)*sin(q1)) - cos(q4)*sin(q1)*sin(q2)))/500 + (39*sin(q6)*(cos(q5)*(cos(q4)*(cos(q1)*sin(q3) + cos(q2)*cos(q3)*sin(q1)) + sin(q1)*sin(q2)*sin(q4)) + sin(q5)*(cos(q1)*cos(q3) - cos(q2)*sin(q1)*sin(q3))))/500 - (39*sin(q4)*(cos(q1)*sin(q3) + cos(q2)*cos(q3)*sin(q1)))/100 + (39*cos(q4)*sin(q1)*sin(q2))/100;
	jacobian(0,1) = - (2*cos(q1)*cos(q2))/5 - (39*cos(q6)*(cos(q1)*cos(q2)*cos(q4) + cos(q1)*cos(q3)*sin(q2)*sin(q4)))/500 - (39*sin(q6)*(cos(q5)*(cos(q1)*cos(q2)*sin(q4) - cos(q1)*cos(q3)*cos(q4)*sin(q2)) + cos(q1)*sin(q2)*sin(q3)*sin(q5)))/500 - (39*cos(q1)*cos(q2)*cos(q4))/100 - (39*cos(q1)*cos(q3)*sin(q2)*sin(q4))/100;
	jacobian(0,2) = - (39*sin(q4)*(cos(q3)*sin(q1) + cos(q1)*cos(q2)*sin(q3)))/100 - (39*sin(q6)*(sin(q5)*(sin(q1)*sin(q3) - cos(q1)*cos(q2)*cos(q3)) - cos(q4)*cos(q5)*(cos(q3)*sin(q1) + cos(q1)*cos(q2)*sin(q3))))/500 - (39*cos(q6)*sin(q4)*(cos(q3)*sin(q1) + cos(q1)*cos(q2)*sin(q3)))/500;
	jacobian(0,3) = (39*cos(q1)*sin(q2)*sin(q4))/100 - (39*cos(q4)*(sin(q1)*sin(q3) - cos(q1)*cos(q2)*cos(q3)))/100 - (39*cos(q5)*sin(q6)*(sin(q4)*(sin(q1)*sin(q3) - cos(q1)*cos(q2)*cos(q3)) + cos(q1)*cos(q4)*sin(q2)))/500 - (39*cos(q6)*(cos(q4)*(sin(q1)*sin(q3) - cos(q1)*cos(q2)*cos(q3)) - cos(q1)*sin(q2)*sin(q4)))/500;
	jacobian(0,4) = -(39*sin(q6)*(sin(q5)*(cos(q4)*(sin(q1)*sin(q3) - cos(q1)*cos(q2)*cos(q3)) - cos(q1)*sin(q2)*sin(q4)) - cos(q5)*(cos(q3)*sin(q1) + cos(q1)*cos(q2)*sin(q3))))/500;
	jacobian(0,5) = (39*sin(q6)*(sin(q4)*(sin(q1)*sin(q3) - cos(q1)*cos(q2)*cos(q3)) + cos(q1)*cos(q4)*sin(q2)))/500 + (39*cos(q6)*(cos(q5)*(cos(q4)*(sin(q1)*sin(q3) - cos(q1)*cos(q2)*cos(q3)) - cos(q1)*sin(q2)*sin(q4)) + sin(q5)*(cos(q3)*sin(q1) + cos(q1)*cos(q2)*sin(q3))))/500;
	jacobian(0,6) = 0;

	jacobian(1,0) = (39*sin(q6)*(cos(q5)*(cos(q4)*(sin(q1)*sin(q3) - cos(q1)*cos(q2)*cos(q3)) - cos(q1)*sin(q2)*sin(q4)) + sin(q5)*(cos(q3)*sin(q1) + cos(q1)*cos(q2)*sin(q3))))/500 - (39*cos(q6)*(sin(q4)*(sin(q1)*sin(q3) - cos(q1)*cos(q2)*cos(q3)) + cos(q1)*cos(q4)*sin(q2)))/500 - (2*cos(q1)*sin(q2))/5 - (39*sin(q4)*(sin(q1)*sin(q3) - cos(q1)*cos(q2)*cos(q3)))/100 - (39*cos(q1)*cos(q4)*sin(q2))/100;
	jacobian(1,1) = - (2*cos(q2)*sin(q1))/5 - (39*cos(q6)*(cos(q2)*cos(q4)*sin(q1) + cos(q3)*sin(q1)*sin(q2)*sin(q4)))/500 - (39*sin(q6)*(cos(q5)*(cos(q2)*sin(q1)*sin(q4) - cos(q3)*cos(q4)*sin(q1)*sin(q2)) + sin(q1)*sin(q2)*sin(q3)*sin(q5)))/500 - (39*cos(q2)*cos(q4)*sin(q1))/100 - (39*cos(q3)*sin(q1)*sin(q2)*sin(q4))/100;
	jacobian(1,2) = (39*sin(q4)*(cos(q1)*cos(q3) - cos(q2)*sin(q1)*sin(q3)))/100 + (39*sin(q6)*(sin(q5)*(cos(q1)*sin(q3) + cos(q2)*cos(q3)*sin(q1)) - cos(q4)*cos(q5)*(cos(q1)*cos(q3) - cos(q2)*sin(q1)*sin(q3))))/500 + (39*cos(q6)*sin(q4)*(cos(q1)*cos(q3) - cos(q2)*sin(q1)*sin(q3)))/500;
	jacobian(1,3) = (39*cos(q6)*(cos(q4)*(cos(q1)*sin(q3) + cos(q2)*cos(q3)*sin(q1)) + sin(q1)*sin(q2)*sin(q4)))/500 + (39*cos(q4)*(cos(q1)*sin(q3) + cos(q2)*cos(q3)*sin(q1)))/100 + (39*sin(q1)*sin(q2)*sin(q4))/100 + (39*cos(q5)*sin(q6)*(sin(q4)*(cos(q1)*sin(q3) + cos(q2)*cos(q3)*sin(q1)) - cos(q4)*sin(q1)*sin(q2)))/500;
	jacobian(1,4) = (39*sin(q6)*(sin(q5)*(cos(q4)*(cos(q1)*sin(q3) + cos(q2)*cos(q3)*sin(q1)) + sin(q1)*sin(q2)*sin(q4)) - cos(q5)*(cos(q1)*cos(q3) - cos(q2)*sin(q1)*sin(q3))))/500;
	jacobian(1,5) = - (39*sin(q6)*(sin(q4)*(cos(q1)*sin(q3) + cos(q2)*cos(q3)*sin(q1)) - cos(q4)*sin(q1)*sin(q2)))/500 - (39*cos(q6)*(cos(q5)*(cos(q4)*(cos(q1)*sin(q3) + cos(q2)*cos(q3)*sin(q1)) + sin(q1)*sin(q2)*sin(q4)) + sin(q5)*(cos(q1)*cos(q3) - cos(q2)*sin(q1)*sin(q3))))/500;
	jacobian(1,6) = 0;

	jacobian(2,0) = 0;
	jacobian(2,1) = (39*cos(q2)*cos(q3)*sin(q4))/100 - (39*cos(q4)*sin(q2))/100 - (39*sin(q6)*(cos(q5)*(sin(q2)*sin(q4) + cos(q2)*cos(q3)*cos(q4)) - cos(q2)*sin(q3)*sin(q5)))/500 - (39*cos(q6)*(cos(q4)*sin(q2) - cos(q2)*cos(q3)*sin(q4)))/500 - (2*sin(q2))/5;
	jacobian(2,2) = (39*sin(q6)*(cos(q3)*sin(q2)*sin(q5) + cos(q4)*cos(q5)*sin(q2)*sin(q3)))/500 - (39*sin(q2)*sin(q3)*sin(q4))/100 - (39*cos(q6)*sin(q2)*sin(q3)*sin(q4))/500;
	jacobian(2,3) = (39*cos(q5)*sin(q6)*(cos(q2)*cos(q4) + cos(q3)*sin(q2)*sin(q4)))/500 - (39*cos(q6)*(cos(q2)*sin(q4) - cos(q3)*cos(q4)*sin(q2)))/500 - (39*cos(q2)*sin(q4))/100 + (39*cos(q3)*cos(q4)*sin(q2))/100;
	jacobian(2,4) = -(39*sin(q6)*(sin(q5)*(cos(q2)*sin(q4) - cos(q3)*cos(q4)*sin(q2)) - cos(q5)*sin(q2)*sin(q3)))/500;
	jacobian(2,5) = (39*cos(q6)*(cos(q5)*(cos(q2)*sin(q4) - cos(q3)*cos(q4)*sin(q2)) + sin(q2)*sin(q3)*sin(q5)))/500 - (39*sin(q6)*(cos(q2)*cos(q4) + cos(q3)*sin(q2)*sin(q4)))/500;
	jacobian(2,6) = 0;

	//cout << "q:" << jacobian(0,0)<< endl; 
	return jacobian;
}

MatrixXd getEndEffectorVelocity(MatrixXd& ys, MatrixXd& yds, VectorXd& ts) {
	double q1 = ys(0), q2 = ys(1), q3 = ys(2), q4 = ys(3), q5 = ys(4), q6 = ys(5), q7 = ys(6);
	MatrixXd jacobian(3,7);  	//jacobian has 3 rows (3 cartesian dimenstions) and 7 rows (7 joints)
	MatrixXd velocity(ts.size(),3);		//x,y,z velocity for all times
	VectorXd temp;                      //just to store temporary values in for loop
	VectorXd q;

	for(int t = 0; t < ts.size(); t++) {
		q = ys.block(t,0,1,7);
		jacobian = getJacobian(q);
		temp = jacobian*yds.row(t).transpose();
		velocity(t,0) = temp[0];
		velocity(t,1) = temp[1];
		velocity(t,2) = temp[2];
	}
	return velocity;
}

//returns the same result as the matlab function pinv, it uses SVD
MatrixXd getPseudoInverse(MatrixXd& J) {
	MatrixXd result;

	JacobiSVD<MatrixXd> svd(J, ComputeFullU | ComputeFullV);
	MatrixXd U = svd.matrixU();
	MatrixXd V = svd.matrixV();

	MatrixXd S = svd.singularValues().asDiagonal();
	MatrixXd Sfull = MatrixXd::Zero(3,7);

	double x = 5;		//tolerance
	for(int i = 0; i < S.rows(); i++) {
		if(S.maxCoeff() < x*S(i,i) && abs(S(i,i)) > 1e-5) {
			Sfull(i,i) = 1/S(i,i);
		}
	}
	MatrixXd Sp = Sfull.transpose();

	result = V*Sp*U.transpose();
	return result;
}


MatrixXd J1(VectorXd& q) {
	MatrixXd j = MatrixXd::Zero(3,7); //3 cartesian dimensions, 7 joints
	double q1 = q[0], q2 = q[1], q3 = q[2], q4 = q[3], q5 = q[4], q6 = q[5], q7 = q[6];
	j(0,0) = 0.0274*c*cos(q1) - 1.0*b*cos(q1) - 1.0*a*sin(q1);
	j(1,0) =  a*cos(q1) - 1.0*b*sin(q1) + 0.0274*c*sin(q1);

	return j;
}

MatrixXd J2(VectorXd& q) {
	MatrixXd j = MatrixXd::Zero(3,7); //3 cartesian dimensions, 7 joints
	double q1 = q[0], q2 = q[1], q3 = q[2], q4 = q[3], q5 = q[4], q6 = q[5], q7 = q[6];

	j(0,0) = c*(1.68e-18*cos(q1) - 0.0274*cos(q1)*cos(q2) + 0.0274*sin(q1)*sin(q2)) - 1.0*a*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) - 1.0*b*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2));
	j(0,1) = - 1.0*c*(0.0274*cos(q1)*cos(q2) - 0.0274*sin(q1)*sin(q2)) - 1.0*b*(1.0*cos(q1)*cos(q2) - 0.999*sin(q1)*sin(q2)) - 1.0*a*(cos(q1)*sin(q2) + 1.0*cos(q2)*sin(q1));


	j(1,0) = a*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) - 1.0*b*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1)) - 1.0*c*(0.0274*cos(q1)*sin(q2) - 1.68e-18*sin(q1) + 0.0274*cos(q2)*sin(q1));
	j(1,1) = a*(1.0*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) - 1.0*c*(0.0274*cos(q1)*sin(q2) + 0.0274*cos(q2)*sin(q1)) - 1.0*b*(0.999*cos(q1)*sin(q2) + 1.0*cos(q2)*sin(q1));

	j(2,1) = a*cos(q2) - 1.0*b*sin(q2) - 0.0274*c*sin(q2);
	//the rest is zero
	return j;
}

MatrixXd J3(VectorXd& q) {
	MatrixXd j = MatrixXd::Zero(3,7); //3 cartesian dimensions, 7 joints
	double q1 = q[0], q2 = q[1], q3 = q[2], q4 = q[3], q5 = q[4], q6 = q[5], q7 = q[6];

	j(0,0) = 6.71e-19*cos(q1) - 0.011*cos(q1)*cos(q2) + 0.011*sin(q1)*sin(q2) + c*(1.03e-34*cos(q1) - 1.68e-18*cos(q1)*cos(q2) + 1.68e-18*sin(q1)*sin(q2) + 0.0274*sin(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) - 0.0274*cos(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2))) - 1.0*b*(1.68e-18*cos(q1) - 0.0274*cos(q1)*cos(q2) + 0.0274*sin(q1)*sin(q2) - 1.0*sin(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) + 1.0*cos(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2))) - 1.0*a*(cos(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) + sin(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)));
	j(0,1) = 0.011*sin(q1)*sin(q2) - 0.011*cos(q1)*cos(q2) - 1.0*a*(cos(q3)*(cos(q1)*sin(q2) + 1.0*cos(q2)*sin(q1)) + sin(q3)*(1.0*cos(q1)*cos(q2) - 0.999*sin(q1)*sin(q2))) - 1.0*c*(1.68e-18*cos(q1)*cos(q2) - 1.68e-18*sin(q1)*sin(q2) - 0.0274*sin(q3)*(cos(q1)*sin(q2) + 1.0*cos(q2)*sin(q1)) + 0.0274*cos(q3)*(1.0*cos(q1)*cos(q2) - 0.999*sin(q1)*sin(q2))) + b*(0.0274*cos(q1)*cos(q2) - 0.0274*sin(q1)*sin(q2) + 1.0*sin(q3)*(cos(q1)*sin(q2) + 1.0*cos(q2)*sin(q1)) - 1.0*cos(q3)*(1.0*cos(q1)*cos(q2) - 0.999*sin(q1)*sin(q2)));
	j(0,2) =  - c*(0.0274*cos(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) - 0.0274*sin(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1))) - 1.0*a*(sin(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) + cos(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1))) - b*(1.0*cos(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) - 1.0*sin(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1)));

	j(1,0) = 6.71e-19*sin(q1) - 0.011*cos(q1)*sin(q2) - 0.011*cos(q2)*sin(q1) + 1.0*a*(1.0*cos(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) - sin(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1))) - 1.0*c*(1.68e-18*cos(q1)*sin(q2) - 1.03e-34*sin(q1) + 1.68e-18*cos(q2)*sin(q1) + 0.0274*sin(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) + 0.0274*cos(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1))) - 1.0*b*(1.68e-18*sin(q1) - 0.0274*cos(q1)*sin(q2) - 0.0274*cos(q2)*sin(q1) + 1.0*sin(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) + 1.0*cos(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1)));
	j(1,1) = 1.0*b*(0.0274*cos(q1)*sin(q2) + 0.0274*cos(q2)*sin(q1) - 1.0*sin(q3)*(1.0*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) - 1.0*cos(q3)*(0.999*cos(q1)*sin(q2) + 1.0*cos(q2)*sin(q1))) - 0.011*cos(q2)*sin(q1) - 1.0*c*(1.68e-18*cos(q1)*sin(q2) + 1.68e-18*cos(q2)*sin(q1) + 0.0274*sin(q3)*(1.0*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) + 0.0274*cos(q3)*(0.999*cos(q1)*sin(q2) + 1.0*cos(q2)*sin(q1))) - 0.011*cos(q1)*sin(q2) + 1.0*a*(1.0*cos(q3)*(1.0*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) - sin(q3)*(0.999*cos(q1)*sin(q2) + 1.0*cos(q2)*sin(q1)));
	j(1,2) = - a*(1.0*sin(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) - cos(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2))) - 1.0*c*(0.0274*cos(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) + 0.0274*sin(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2))) - 1.0*b*(1.0*cos(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) + 1.0*sin(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)));

	j(2,1) = a*(cos(q2)*cos(q3) - 1.0*sin(q2)*sin(q3)) - 1.0*b*(1.0*cos(q2)*sin(q3) - 0.0274*sin(q2) + 0.999*cos(q3)*sin(q2)) - 0.011*sin(q2) - 1.0*c*(1.68e-18*sin(q2) + 0.0274*cos(q2)*sin(q3) + 0.0274*cos(q3)*sin(q2));
	j(2,2) = - 1.0*a*(sin(q2)*sin(q3) - 1.0*cos(q3)*(1.0*cos(q2) - 6.12e-17)) - 1.0*c*(0.0274*cos(q3)*sin(q2) + 0.0274*sin(q3)*(1.0*cos(q2) - 6.12e-17)) - 1.0*b*(1.0*cos(q3)*sin(q2) + 1.0*sin(q3)*(1.0*cos(q2) - 6.12e-17));

	return j;
}

MatrixXd J4(VectorXd& q) {
	MatrixXd j = MatrixXd::Zero(3,7); //3 cartesian dimensions, 7 joints
	double q1 = q[0], q2 = q[1], q3 = q[2], q4 = q[3], q5 = q[4], q6 = q[5], q7 = q[6];

	j(0,0) = 6.71e-19*cos(q1) - 0.011*cos(q1)*cos(q2) + 0.011*sin(q1)*sin(q2) + c*(6.29e-51*cos(q1) - 1.03e-34*cos(q1)*cos(q2) + 0.0274*cos(q4)*(1.68e-18*cos(q1) - 0.0274*cos(q1)*cos(q2) + 0.0274*sin(q1)*sin(q2) - 1.0*sin(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) + 1.0*cos(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2))) + 1.03e-34*sin(q1)*sin(q2) + 1.68e-18*sin(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) - 0.0274*sin(q4)*(cos(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) + sin(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2))) - 1.68e-18*cos(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2))) - 1.0*a*(sin(q4)*(1.68e-18*cos(q1) - 0.0274*cos(q1)*cos(q2) + 0.0274*sin(q1)*sin(q2) - 1.0*sin(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) + 1.0*cos(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2))) + cos(q4)*(cos(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) + sin(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)))) + b*(1.03e-34*cos(q1) - 1.68e-18*cos(q1)*cos(q2) - 1.0*cos(q4)*(1.68e-18*cos(q1) - 0.0274*cos(q1)*cos(q2) + 0.0274*sin(q1)*sin(q2) - 1.0*sin(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) + 1.0*cos(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2))) + 1.68e-18*sin(q1)*sin(q2) + 0.0274*sin(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) + 1.0*sin(q4)*(cos(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) + sin(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2))) - 0.0274*cos(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)));
	j(0,1) = 0.011*sin(q1)*sin(q2) - 0.011*cos(q1)*cos(q2) - 1.0*c*(1.03e-34*cos(q1)*cos(q2) - 1.03e-34*sin(q1)*sin(q2) - 1.68e-18*sin(q3)*(cos(q1)*sin(q2) + 1.0*cos(q2)*sin(q1)) + 0.0274*sin(q4)*(cos(q3)*(cos(q1)*sin(q2) + 1.0*cos(q2)*sin(q1)) + sin(q3)*(1.0*cos(q1)*cos(q2) - 0.999*sin(q1)*sin(q2))) + 0.0274*cos(q4)*(0.0274*cos(q1)*cos(q2) - 0.0274*sin(q1)*sin(q2) + 1.0*sin(q3)*(cos(q1)*sin(q2) + 1.0*cos(q2)*sin(q1)) - 1.0*cos(q3)*(1.0*cos(q1)*cos(q2) - 0.999*sin(q1)*sin(q2))) + 1.68e-18*cos(q3)*(1.0*cos(q1)*cos(q2) - 0.999*sin(q1)*sin(q2))) + b*(1.68e-18*sin(q1)*sin(q2) - 1.68e-18*cos(q1)*cos(q2) + 0.0274*sin(q3)*(cos(q1)*sin(q2) + 1.0*cos(q2)*sin(q1)) + 1.0*sin(q4)*(cos(q3)*(cos(q1)*sin(q2) + 1.0*cos(q2)*sin(q1)) + sin(q3)*(1.0*cos(q1)*cos(q2) - 0.999*sin(q1)*sin(q2))) + 1.0*cos(q4)*(0.0274*cos(q1)*cos(q2) - 0.0274*sin(q1)*sin(q2) + 1.0*sin(q3)*(cos(q1)*sin(q2) + 1.0*cos(q2)*sin(q1)) - 1.0*cos(q3)*(1.0*cos(q1)*cos(q2) - 0.999*sin(q1)*sin(q2))) - 0.0274*cos(q3)*(1.0*cos(q1)*cos(q2) - 0.999*sin(q1)*sin(q2))) - 1.0*a*(cos(q4)*(cos(q3)*(cos(q1)*sin(q2) + 1.0*cos(q2)*sin(q1)) + sin(q3)*(1.0*cos(q1)*cos(q2) - 0.999*sin(q1)*sin(q2))) - 1.0*sin(q4)*(0.0274*cos(q1)*cos(q2) - 0.0274*sin(q1)*sin(q2) + 1.0*sin(q3)*(cos(q1)*sin(q2) + 1.0*cos(q2)*sin(q1)) - 1.0*cos(q3)*(1.0*cos(q1)*cos(q2) - 0.999*sin(q1)*sin(q2))));
	j(0,2) = - 1.0*a*(1.0*cos(q4)*(sin(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) + cos(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1))) + sin(q4)*(1.0*cos(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) - 1.0*sin(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1)))) - 1.0*c*(1.68e-18*cos(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) - 0.0274*cos(q4)*(1.0*cos(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) - 1.0*sin(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1))) + 0.0274*sin(q4)*(sin(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) + cos(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1))) - 1.68e-18*sin(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1))) - 1.0*b*(0.0274*cos(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) + 1.0*cos(q4)*(1.0*cos(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) - 1.0*sin(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1))) - 1.0*sin(q4)*(sin(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) + cos(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1))) - 0.0274*sin(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1)));
	j(0,3) = b*(1.0*sin(q4)*(1.68e-18*sin(q1) - 0.0274*cos(q1)*sin(q2) - 0.0274*cos(q2)*sin(q1) + 1.0*sin(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) + 1.0*cos(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1))) - 1.0*cos(q4)*(1.0*cos(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) - 1.0*sin(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1)))) - 1.0*c*(0.0274*sin(q4)*(1.68e-18*sin(q1) - 0.0274*cos(q1)*sin(q2) - 0.0274*cos(q2)*sin(q1) + 1.0*sin(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) + 1.0*cos(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1))) - 0.0274*cos(q4)*(1.0*cos(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) - 1.0*sin(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1)))) - 1.0*a*(cos(q4)*(1.68e-18*sin(q1) - 0.0274*cos(q1)*sin(q2) - 0.0274*cos(q2)*sin(q1) + 1.0*sin(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) + 1.0*cos(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1))) + 1.0*sin(q4)*(1.0*cos(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) - 1.0*sin(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1))));

	j(1,0) = 6.71e-19*sin(q1) - 0.011*cos(q1)*sin(q2) - 0.011*cos(q2)*sin(q1) - 1.0*b*(1.68e-18*cos(q1)*sin(q2) - 1.03e-34*sin(q1) + 1.68e-18*cos(q2)*sin(q1) + 1.0*cos(q4)*(1.68e-18*sin(q1) - 0.0274*cos(q1)*sin(q2) - 0.0274*cos(q2)*sin(q1) + 1.0*sin(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) + 1.0*cos(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1))) + 0.0274*sin(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) + 0.0274*cos(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1)) + 1.0*sin(q4)*(1.0*cos(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) - sin(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1)))) - 1.0*c*(1.03e-34*cos(q1)*sin(q2) - 6.29e-51*sin(q1) + 1.03e-34*cos(q2)*sin(q1) - 0.0274*cos(q4)*(1.68e-18*sin(q1) - 0.0274*cos(q1)*sin(q2) - 0.0274*cos(q2)*sin(q1) + 1.0*sin(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) + 1.0*cos(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1))) + 1.68e-18*sin(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) + 1.68e-18*cos(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1)) - 0.0274*sin(q4)*(1.0*cos(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) - sin(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1)))) - 1.0*a*(sin(q4)*(1.68e-18*sin(q1) - 0.0274*cos(q1)*sin(q2) - 0.0274*cos(q2)*sin(q1) + 1.0*sin(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) + 1.0*cos(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1))) - cos(q4)*(1.0*cos(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) - sin(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1))));
	j(1,1) = 1.0*a*(sin(q4)*(0.0274*cos(q1)*sin(q2) + 0.0274*cos(q2)*sin(q1) - 1.0*sin(q3)*(1.0*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) - 1.0*cos(q3)*(0.999*cos(q1)*sin(q2) + 1.0*cos(q2)*sin(q1))) + cos(q4)*(1.0*cos(q3)*(1.0*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) - sin(q3)*(0.999*cos(q1)*sin(q2) + 1.0*cos(q2)*sin(q1)))) - 0.011*cos(q2)*sin(q1) - 1.0*c*(1.03e-34*cos(q1)*sin(q2) + 1.03e-34*cos(q2)*sin(q1) - 0.0274*sin(q4)*(1.0*cos(q3)*(1.0*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) - sin(q3)*(0.999*cos(q1)*sin(q2) + 1.0*cos(q2)*sin(q1))) + 1.68e-18*sin(q3)*(1.0*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) + 1.68e-18*cos(q3)*(0.999*cos(q1)*sin(q2) + 1.0*cos(q2)*sin(q1)) + 0.0274*cos(q4)*(0.0274*cos(q1)*sin(q2) + 0.0274*cos(q2)*sin(q1) - 1.0*sin(q3)*(1.0*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) - 1.0*cos(q3)*(0.999*cos(q1)*sin(q2) + 1.0*cos(q2)*sin(q1)))) - 0.011*cos(q1)*sin(q2) - 1.0*b*(1.68e-18*cos(q1)*sin(q2) + 1.68e-18*cos(q2)*sin(q1) + 1.0*sin(q4)*(1.0*cos(q3)*(1.0*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) - sin(q3)*(0.999*cos(q1)*sin(q2) + 1.0*cos(q2)*sin(q1))) + 0.0274*sin(q3)*(1.0*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) + 0.0274*cos(q3)*(0.999*cos(q1)*sin(q2) + 1.0*cos(q2)*sin(q1)) - 1.0*cos(q4)*(0.0274*cos(q1)*sin(q2) + 0.0274*cos(q2)*sin(q1) - 1.0*sin(q3)*(1.0*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) - 1.0*cos(q3)*(0.999*cos(q1)*sin(q2) + 1.0*cos(q2)*sin(q1))));
	j(1,2) = - 1.0*a*(sin(q4)*(1.0*cos(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) + 1.0*sin(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2))) + 1.0*cos(q4)*(1.0*sin(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) - cos(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)))) - 1.0*c*(1.68e-18*cos(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) - 0.0274*cos(q4)*(1.0*cos(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) + 1.0*sin(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2))) + 1.68e-18*sin(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) + 0.0274*sin(q4)*(1.0*sin(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) - cos(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)))) - 1.0*b*(0.0274*cos(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) + 1.0*cos(q4)*(1.0*cos(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) + 1.0*sin(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2))) + 0.0274*sin(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) - 1.0*sin(q4)*(1.0*sin(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) - cos(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2))));
	j(1,3) = c*(0.0274*sin(q4)*(1.68e-18*cos(q1) - 0.0274*cos(q1)*cos(q2) + 0.0274*sin(q1)*sin(q2) - 1.0*sin(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) + 1.0*cos(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2))) + 0.0274*cos(q4)*(cos(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) + sin(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)))) - 1.0*b*(1.0*sin(q4)*(1.68e-18*cos(q1) - 0.0274*cos(q1)*cos(q2) + 0.0274*sin(q1)*sin(q2) - 1.0*sin(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) + 1.0*cos(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2))) + 1.0*cos(q4)*(cos(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) + sin(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)))) + a*(cos(q4)*(1.68e-18*cos(q1) - 0.0274*cos(q1)*cos(q2) + 0.0274*sin(q1)*sin(q2) - 1.0*sin(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) + 1.0*cos(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2))) - 1.0*sin(q4)*(cos(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) + sin(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2))));

	j(2,1) = 1.0*a*(1.0*cos(q4)*(cos(q2)*cos(q3) - 1.0*sin(q2)*sin(q3)) - sin(q4)*(1.0*cos(q2)*sin(q3) - 0.0274*sin(q2) + 0.999*cos(q3)*sin(q2))) - 0.011*sin(q2) - 1.0*c*(1.03e-34*sin(q2) + 1.68e-18*cos(q2)*sin(q3) + 1.68e-18*cos(q3)*sin(q2) - 0.0274*sin(q4)*(cos(q2)*cos(q3) - 1.0*sin(q2)*sin(q3)) - 0.0274*cos(q4)*(1.0*cos(q2)*sin(q3) - 0.0274*sin(q2) + 0.999*cos(q3)*sin(q2))) - 1.0*b*(1.68e-18*sin(q2) + 0.0274*cos(q2)*sin(q3) + 0.0274*cos(q3)*sin(q2) + 1.0*sin(q4)*(cos(q2)*cos(q3) - 1.0*sin(q2)*sin(q3)) + 1.0*cos(q4)*(1.0*cos(q2)*sin(q3) - 0.0274*sin(q2) + 0.999*cos(q3)*sin(q2)));
	j(2,2) = - 1.0*a*(sin(q4)*(1.0*cos(q3)*sin(q2) + 1.0*sin(q3)*(1.0*cos(q2) - 6.12e-17)) + cos(q4)*(sin(q2)*sin(q3) - 1.0*cos(q3)*(1.0*cos(q2) - 6.12e-17))) - 1.0*c*(1.68e-18*cos(q3)*sin(q2) - 0.0274*cos(q4)*(1.0*cos(q3)*sin(q2) + 1.0*sin(q3)*(1.0*cos(q2) - 6.12e-17)) + 0.0274*sin(q4)*(sin(q2)*sin(q3) - 1.0*cos(q3)*(1.0*cos(q2) - 6.12e-17)) + 1.68e-18*sin(q3)*(1.0*cos(q2) - 6.12e-17)) - 1.0*b*(0.0274*cos(q3)*sin(q2) + 1.0*cos(q4)*(1.0*cos(q3)*sin(q2) + 1.0*sin(q3)*(1.0*cos(q2) - 6.12e-17)) - 1.0*sin(q4)*(sin(q2)*sin(q3) - 1.0*cos(q3)*(1.0*cos(q2) - 6.12e-17)) + 0.0274*sin(q3)*(1.0*cos(q2) - 6.12e-17));
	j(2,3) = 1.0*c*(0.0274*cos(q4)*(cos(q3)*sin(q2) + sin(q3)*(1.0*cos(q2) - 6.12e-17)) - 0.0274*sin(q4)*(0.0274*cos(q2) + 1.0*sin(q2)*sin(q3) - 1.0*cos(q3)*(1.0*cos(q2) - 6.12e-17) + 3.75e-33)) - 1.0*a*(cos(q4)*(0.0274*cos(q2) + 1.0*sin(q2)*sin(q3) - 1.0*cos(q3)*(1.0*cos(q2) - 6.12e-17) + 3.75e-33) + sin(q4)*(cos(q3)*sin(q2) + sin(q3)*(1.0*cos(q2) - 6.12e-17))) - b*(1.0*cos(q4)*(cos(q3)*sin(q2) + sin(q3)*(1.0*cos(q2) - 6.12e-17)) - 1.0*sin(q4)*(0.0274*cos(q2) + 1.0*sin(q2)*sin(q3) - 1.0*cos(q3)*(1.0*cos(q2) - 6.12e-17) + 3.75e-33));

	return j;
}

MatrixXd J5(VectorXd& q) {
	MatrixXd j = MatrixXd::Zero(3,7); //3 cartesian dimensions, 7 joints
	double q1 = q[0], q2 = q[1], q3 = q[2], q4 = q[3], q5 = q[4], q6 = q[5], q7 = q[6];

	j(0,0) = 6.71e-19*cos(q1) - 0.011*cos(q1)*cos(q2) + 0.0107*cos(q4)*(1.68e-18*cos(q1) - 0.0274*cos(q1)*cos(q2) + 0.0274*sin(q1)*sin(q2) - 1.0*sin(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) + 1.0*cos(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2))) + 0.011*sin(q1)*sin(q2) - 1.0*c*(6.29e-51*cos(q1)*cos(q2) - 3.85e-67*cos(q1) - 1.68e-18*cos(q4)*(1.68e-18*cos(q1) - 0.0274*cos(q1)*cos(q2) + 0.0274*sin(q1)*sin(q2) - 1.0*sin(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) + 1.0*cos(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2))) + 0.0274*sin(q5)*(sin(q4)*(1.68e-18*cos(q1) - 0.0274*cos(q1)*cos(q2) + 0.0274*sin(q1)*sin(q2) - 1.0*sin(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) + 1.0*cos(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2))) + cos(q4)*(cos(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) + sin(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)))) - 6.29e-51*sin(q1)*sin(q2) + 0.0274*cos(q5)*(1.03e-34*cos(q1) - 1.68e-18*cos(q1)*cos(q2) - 1.0*cos(q4)*(1.68e-18*cos(q1) - 0.0274*cos(q1)*cos(q2) + 0.0274*sin(q1)*sin(q2) - 1.0*sin(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) + 1.0*cos(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2))) + 1.68e-18*sin(q1)*sin(q2) + 0.0274*sin(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) + 1.0*sin(q4)*(cos(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) + sin(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2))) - 0.0274*cos(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2))) - 1.03e-34*sin(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) + 1.68e-18*sin(q4)*(cos(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) + sin(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2))) + 1.03e-34*cos(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2))) + 6.55e-19*sin(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) - 0.0107*sin(q4)*(cos(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) + sin(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2))) - 6.55e-19*cos(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) + b*(6.29e-51*cos(q1) - 1.03e-34*cos(q1)*cos(q2) + 0.0274*cos(q4)*(1.68e-18*cos(q1) - 0.0274*cos(q1)*cos(q2) + 0.0274*sin(q1)*sin(q2) - 1.0*sin(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) + 1.0*cos(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2))) + 1.0*sin(q5)*(sin(q4)*(1.68e-18*cos(q1) - 0.0274*cos(q1)*cos(q2) + 0.0274*sin(q1)*sin(q2) - 1.0*sin(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) + 1.0*cos(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2))) + cos(q4)*(cos(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) + sin(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)))) + 1.03e-34*sin(q1)*sin(q2) + 1.0*cos(q5)*(1.03e-34*cos(q1) - 1.68e-18*cos(q1)*cos(q2) - 1.0*cos(q4)*(1.68e-18*cos(q1) - 0.0274*cos(q1)*cos(q2) + 0.0274*sin(q1)*sin(q2) - 1.0*sin(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) + 1.0*cos(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2))) + 1.68e-18*sin(q1)*sin(q2) + 0.0274*sin(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) + 1.0*sin(q4)*(cos(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) + sin(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2))) - 0.0274*cos(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2))) + 1.68e-18*sin(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) - 0.0274*sin(q4)*(cos(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) + sin(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2))) - 1.68e-18*cos(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2))) - 1.0*a*(1.0*cos(q5)*(sin(q4)*(1.68e-18*cos(q1) - 0.0274*cos(q1)*cos(q2) + 0.0274*sin(q1)*sin(q2) - 1.0*sin(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) + 1.0*cos(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2))) + cos(q4)*(cos(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) + sin(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)))) - 1.0*sin(q5)*(1.03e-34*cos(q1) - 1.68e-18*cos(q1)*cos(q2) - 1.0*cos(q4)*(1.68e-18*cos(q1) - 0.0274*cos(q1)*cos(q2) + 0.0274*sin(q1)*sin(q2) - 1.0*sin(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) + 1.0*cos(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2))) + 1.68e-18*sin(q1)*sin(q2) + 0.0274*sin(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) + 1.0*sin(q4)*(cos(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) + sin(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2))) - 0.0274*cos(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2))));
	j(0,1) = 0.011*sin(q1)*sin(q2) - 0.011*cos(q1)*cos(q2) + 6.55e-19*sin(q3)*(cos(q1)*sin(q2) + 1.0*cos(q2)*sin(q1)) - 1.0*c*(6.29e-51*cos(q1)*cos(q2) - 6.29e-51*sin(q1)*sin(q2) + 0.0274*cos(q5)*(1.68e-18*sin(q1)*sin(q2) - 1.68e-18*cos(q1)*cos(q2) + 0.0274*sin(q3)*(cos(q1)*sin(q2) + 1.0*cos(q2)*sin(q1)) + 1.0*sin(q4)*(cos(q3)*(cos(q1)*sin(q2) + 1.0*cos(q2)*sin(q1)) + sin(q3)*(1.0*cos(q1)*cos(q2) - 0.999*sin(q1)*sin(q2))) + 1.0*cos(q4)*(0.0274*cos(q1)*cos(q2) - 0.0274*sin(q1)*sin(q2) + 1.0*sin(q3)*(cos(q1)*sin(q2) + 1.0*cos(q2)*sin(q1)) - 1.0*cos(q3)*(1.0*cos(q1)*cos(q2) - 0.999*sin(q1)*sin(q2))) - 0.0274*cos(q3)*(1.0*cos(q1)*cos(q2) - 0.999*sin(q1)*sin(q2))) - 1.03e-34*sin(q3)*(cos(q1)*sin(q2) + 1.0*cos(q2)*sin(q1)) + 0.0274*sin(q5)*(cos(q4)*(cos(q3)*(cos(q1)*sin(q2) + 1.0*cos(q2)*sin(q1)) + sin(q3)*(1.0*cos(q1)*cos(q2) - 0.999*sin(q1)*sin(q2))) - 1.0*sin(q4)*(0.0274*cos(q1)*cos(q2) - 0.0274*sin(q1)*sin(q2) + 1.0*sin(q3)*(cos(q1)*sin(q2) + 1.0*cos(q2)*sin(q1)) - 1.0*cos(q3)*(1.0*cos(q1)*cos(q2) - 0.999*sin(q1)*sin(q2)))) + 1.68e-18*sin(q4)*(cos(q3)*(cos(q1)*sin(q2) + 1.0*cos(q2)*sin(q1)) + sin(q3)*(1.0*cos(q1)*cos(q2) - 0.999*sin(q1)*sin(q2))) + 1.68e-18*cos(q4)*(0.0274*cos(q1)*cos(q2) - 0.0274*sin(q1)*sin(q2) + 1.0*sin(q3)*(cos(q1)*sin(q2) + 1.0*cos(q2)*sin(q1)) - 1.0*cos(q3)*(1.0*cos(q1)*cos(q2) - 0.999*sin(q1)*sin(q2))) + 1.03e-34*cos(q3)*(1.0*cos(q1)*cos(q2) - 0.999*sin(q1)*sin(q2))) - 0.0107*sin(q4)*(cos(q3)*(cos(q1)*sin(q2) + 1.0*cos(q2)*sin(q1)) + sin(q3)*(1.0*cos(q1)*cos(q2) - 0.999*sin(q1)*sin(q2))) - 0.0107*cos(q4)*(0.0274*cos(q1)*cos(q2) - 0.0274*sin(q1)*sin(q2) + 1.0*sin(q3)*(cos(q1)*sin(q2) + 1.0*cos(q2)*sin(q1)) - 1.0*cos(q3)*(1.0*cos(q1)*cos(q2) - 0.999*sin(q1)*sin(q2))) - 6.55e-19*cos(q3)*(1.0*cos(q1)*cos(q2) - 0.999*sin(q1)*sin(q2)) - 1.0*b*(1.03e-34*cos(q1)*cos(q2) - 1.03e-34*sin(q1)*sin(q2) - 1.0*cos(q5)*(1.68e-18*sin(q1)*sin(q2) - 1.68e-18*cos(q1)*cos(q2) + 0.0274*sin(q3)*(cos(q1)*sin(q2) + 1.0*cos(q2)*sin(q1)) + 1.0*sin(q4)*(cos(q3)*(cos(q1)*sin(q2) + 1.0*cos(q2)*sin(q1)) + sin(q3)*(1.0*cos(q1)*cos(q2) - 0.999*sin(q1)*sin(q2))) + 1.0*cos(q4)*(0.0274*cos(q1)*cos(q2) - 0.0274*sin(q1)*sin(q2) + 1.0*sin(q3)*(cos(q1)*sin(q2) + 1.0*cos(q2)*sin(q1)) - 1.0*cos(q3)*(1.0*cos(q1)*cos(q2) - 0.999*sin(q1)*sin(q2))) - 0.0274*cos(q3)*(1.0*cos(q1)*cos(q2) - 0.999*sin(q1)*sin(q2))) - 1.68e-18*sin(q3)*(cos(q1)*sin(q2) + 1.0*cos(q2)*sin(q1)) - 1.0*sin(q5)*(cos(q4)*(cos(q3)*(cos(q1)*sin(q2) + 1.0*cos(q2)*sin(q1)) + sin(q3)*(1.0*cos(q1)*cos(q2) - 0.999*sin(q1)*sin(q2))) - 1.0*sin(q4)*(0.0274*cos(q1)*cos(q2) - 0.0274*sin(q1)*sin(q2) + 1.0*sin(q3)*(cos(q1)*sin(q2) + 1.0*cos(q2)*sin(q1)) - 1.0*cos(q3)*(1.0*cos(q1)*cos(q2) - 0.999*sin(q1)*sin(q2)))) + 0.0274*sin(q4)*(cos(q3)*(cos(q1)*sin(q2) + 1.0*cos(q2)*sin(q1)) + sin(q3)*(1.0*cos(q1)*cos(q2) - 0.999*sin(q1)*sin(q2))) + 0.0274*cos(q4)*(0.0274*cos(q1)*cos(q2) - 0.0274*sin(q1)*sin(q2) + 1.0*sin(q3)*(cos(q1)*sin(q2) + 1.0*cos(q2)*sin(q1)) - 1.0*cos(q3)*(1.0*cos(q1)*cos(q2) - 0.999*sin(q1)*sin(q2))) + 1.68e-18*cos(q3)*(1.0*cos(q1)*cos(q2) - 0.999*sin(q1)*sin(q2))) + a*(sin(q5)*(1.68e-18*sin(q1)*sin(q2) - 1.68e-18*cos(q1)*cos(q2) + 0.0274*sin(q3)*(cos(q1)*sin(q2) + 1.0*cos(q2)*sin(q1)) + 1.0*sin(q4)*(cos(q3)*(cos(q1)*sin(q2) + 1.0*cos(q2)*sin(q1)) + sin(q3)*(1.0*cos(q1)*cos(q2) - 0.999*sin(q1)*sin(q2))) + 1.0*cos(q4)*(0.0274*cos(q1)*cos(q2) - 0.0274*sin(q1)*sin(q2) + 1.0*sin(q3)*(cos(q1)*sin(q2) + 1.0*cos(q2)*sin(q1)) - 1.0*cos(q3)*(1.0*cos(q1)*cos(q2) - 0.999*sin(q1)*sin(q2))) - 0.0274*cos(q3)*(1.0*cos(q1)*cos(q2) - 0.999*sin(q1)*sin(q2))) - 1.0*cos(q5)*(cos(q4)*(cos(q3)*(cos(q1)*sin(q2) + 1.0*cos(q2)*sin(q1)) + sin(q3)*(1.0*cos(q1)*cos(q2) - 0.999*sin(q1)*sin(q2))) - 1.0*sin(q4)*(0.0274*cos(q1)*cos(q2) - 0.0274*sin(q1)*sin(q2) + 1.0*sin(q3)*(cos(q1)*sin(q2) + 1.0*cos(q2)*sin(q1)) - 1.0*cos(q3)*(1.0*cos(q1)*cos(q2) - 0.999*sin(q1)*sin(q2)))));
	j(0,2) = 0.0107*cos(q4)*(1.0*cos(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) - 1.0*sin(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1))) - 1.0*b*(1.0*cos(q5)*(0.0274*cos(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) + 1.0*cos(q4)*(1.0*cos(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) - 1.0*sin(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1))) - 1.0*sin(q4)*(sin(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) + cos(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1))) - 0.0274*sin(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1))) + 1.68e-18*cos(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) - 0.0274*cos(q4)*(1.0*cos(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) - 1.0*sin(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1))) + 0.0274*sin(q4)*(sin(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) + cos(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1))) - 1.68e-18*sin(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1)) - 1.0*sin(q5)*(1.0*cos(q4)*(sin(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) + cos(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1))) + sin(q4)*(1.0*cos(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) - 1.0*sin(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1))))) - 6.55e-19*cos(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) - 0.0107*sin(q4)*(sin(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) + cos(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1))) + 6.55e-19*sin(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1)) + 1.0*c*(0.0274*cos(q5)*(0.0274*cos(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) + 1.0*cos(q4)*(1.0*cos(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) - 1.0*sin(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1))) - 1.0*sin(q4)*(sin(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) + cos(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1))) - 0.0274*sin(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1))) - 1.03e-34*cos(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) + 1.68e-18*cos(q4)*(1.0*cos(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) - 1.0*sin(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1))) - 1.68e-18*sin(q4)*(sin(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) + cos(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1))) + 1.03e-34*sin(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1)) - 0.0274*sin(q5)*(1.0*cos(q4)*(sin(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) + cos(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1))) + sin(q4)*(1.0*cos(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) - 1.0*sin(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1))))) - 1.0*a*(sin(q5)*(0.0274*cos(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) + 1.0*cos(q4)*(1.0*cos(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) - 1.0*sin(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1))) - 1.0*sin(q4)*(sin(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) + cos(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1))) - 0.0274*sin(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1))) + cos(q5)*(1.0*cos(q4)*(sin(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) + cos(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1))) + sin(q4)*(1.0*cos(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) - 1.0*sin(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1)))));
	j(0,3) = a*(sin(q5)*(1.0*sin(q4)*(1.68e-18*sin(q1) - 0.0274*cos(q1)*sin(q2) - 0.0274*cos(q2)*sin(q1) + 1.0*sin(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) + 1.0*cos(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1))) - 1.0*cos(q4)*(1.0*cos(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) - 1.0*sin(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1)))) - 1.0*cos(q5)*(cos(q4)*(1.68e-18*sin(q1) - 0.0274*cos(q1)*sin(q2) - 0.0274*cos(q2)*sin(q1) + 1.0*sin(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) + 1.0*cos(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1))) + 1.0*sin(q4)*(1.0*cos(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) - 1.0*sin(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1))))) - 0.0107*sin(q4)*(1.68e-18*sin(q1) - 0.0274*cos(q1)*sin(q2) - 0.0274*cos(q2)*sin(q1) + 1.0*sin(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) + 1.0*cos(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1))) - 1.0*c*(0.0274*sin(q5)*(cos(q4)*(1.68e-18*sin(q1) - 0.0274*cos(q1)*sin(q2) - 0.0274*cos(q2)*sin(q1) + 1.0*sin(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) + 1.0*cos(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1))) + 1.0*sin(q4)*(1.0*cos(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) - 1.0*sin(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1)))) + 1.68e-18*sin(q4)*(1.68e-18*sin(q1) - 0.0274*cos(q1)*sin(q2) - 0.0274*cos(q2)*sin(q1) + 1.0*sin(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) + 1.0*cos(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1))) + 0.0274*cos(q5)*(1.0*sin(q4)*(1.68e-18*sin(q1) - 0.0274*cos(q1)*sin(q2) - 0.0274*cos(q2)*sin(q1) + 1.0*sin(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) + 1.0*cos(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1))) - 1.0*cos(q4)*(1.0*cos(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) - 1.0*sin(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1)))) - 1.68e-18*cos(q4)*(1.0*cos(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) - 1.0*sin(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1)))) + 1.0*b*(1.0*sin(q5)*(cos(q4)*(1.68e-18*sin(q1) - 0.0274*cos(q1)*sin(q2) - 0.0274*cos(q2)*sin(q1) + 1.0*sin(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) + 1.0*cos(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1))) + 1.0*sin(q4)*(1.0*cos(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) - 1.0*sin(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1)))) - 0.0274*sin(q4)*(1.68e-18*sin(q1) - 0.0274*cos(q1)*sin(q2) - 0.0274*cos(q2)*sin(q1) + 1.0*sin(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) + 1.0*cos(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1))) + 1.0*cos(q5)*(1.0*sin(q4)*(1.68e-18*sin(q1) - 0.0274*cos(q1)*sin(q2) - 0.0274*cos(q2)*sin(q1) + 1.0*sin(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) + 1.0*cos(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1))) - 1.0*cos(q4)*(1.0*cos(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) - 1.0*sin(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1)))) + 0.0274*cos(q4)*(1.0*cos(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) - 1.0*sin(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1)))) + 0.0107*cos(q4)*(1.0*cos(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) - 1.0*sin(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1)));
	j(0,4) = b*(1.0*sin(q5)*(1.68e-18*cos(q1)*sin(q2) - 1.03e-34*sin(q1) + 1.68e-18*cos(q2)*sin(q1) + 1.0*cos(q4)*(1.68e-18*sin(q1) - 0.0274*cos(q1)*sin(q2) - 0.0274*cos(q2)*sin(q1) + 1.0*sin(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) + 1.0*cos(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1))) + 0.0274*sin(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) + 0.0274*cos(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1)) + 1.0*sin(q4)*(1.0*cos(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) - 1.0*sin(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1)))) + 1.0*cos(q5)*(sin(q4)*(1.68e-18*sin(q1) - 0.0274*cos(q1)*sin(q2) - 0.0274*cos(q2)*sin(q1) + 1.0*sin(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) + 1.0*cos(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1))) - 1.0*cos(q4)*(1.0*cos(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) - 1.0*sin(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1))))) - 1.0*c*(0.0274*sin(q5)*(1.68e-18*cos(q1)*sin(q2) - 1.03e-34*sin(q1) + 1.68e-18*cos(q2)*sin(q1) + 1.0*cos(q4)*(1.68e-18*sin(q1) - 0.0274*cos(q1)*sin(q2) - 0.0274*cos(q2)*sin(q1) + 1.0*sin(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) + 1.0*cos(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1))) + 0.0274*sin(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) + 0.0274*cos(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1)) + 1.0*sin(q4)*(1.0*cos(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) - 1.0*sin(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1)))) + 0.0274*cos(q5)*(sin(q4)*(1.68e-18*sin(q1) - 0.0274*cos(q1)*sin(q2) - 0.0274*cos(q2)*sin(q1) + 1.0*sin(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) + 1.0*cos(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1))) - 1.0*cos(q4)*(1.0*cos(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) - 1.0*sin(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1))))) + 1.0*a*(1.0*sin(q5)*(sin(q4)*(1.68e-18*sin(q1) - 0.0274*cos(q1)*sin(q2) - 0.0274*cos(q2)*sin(q1) + 1.0*sin(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) + 1.0*cos(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1))) - 1.0*cos(q4)*(1.0*cos(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) - 1.0*sin(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1)))) - 1.0*cos(q5)*(1.68e-18*cos(q1)*sin(q2) - 1.03e-34*sin(q1) + 1.68e-18*cos(q2)*sin(q1) + 1.0*cos(q4)*(1.68e-18*sin(q1) - 0.0274*cos(q1)*sin(q2) - 0.0274*cos(q2)*sin(q1) + 1.0*sin(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) + 1.0*cos(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1))) + 0.0274*sin(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) + 0.0274*cos(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1)) + 1.0*sin(q4)*(1.0*cos(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) - 1.0*sin(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1)))));

	j(1,0) = 6.71e-19*sin(q1) - 0.011*cos(q1)*sin(q2) - 0.011*cos(q2)*sin(q1) + 0.0107*cos(q4)*(1.68e-18*sin(q1) - 0.0274*cos(q1)*sin(q2) - 0.0274*cos(q2)*sin(q1) + 1.0*sin(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) + 1.0*cos(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1))) - 1.0*a*(sin(q5)*(1.68e-18*cos(q1)*sin(q2) - 1.03e-34*sin(q1) + 1.68e-18*cos(q2)*sin(q1) + 1.0*cos(q4)*(1.68e-18*sin(q1) - 0.0274*cos(q1)*sin(q2) - 0.0274*cos(q2)*sin(q1) + 1.0*sin(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) + 1.0*cos(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1))) + 0.0274*sin(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) + 0.0274*cos(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1)) + 1.0*sin(q4)*(1.0*cos(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) - 1.0*sin(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1)))) + cos(q5)*(sin(q4)*(1.68e-18*sin(q1) - 0.0274*cos(q1)*sin(q2) - 0.0274*cos(q2)*sin(q1) + 1.0*sin(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) + 1.0*cos(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1))) - 1.0*cos(q4)*(1.0*cos(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) - 1.0*sin(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1))))) - 1.0*b*(1.0*cos(q5)*(1.68e-18*cos(q1)*sin(q2) - 1.03e-34*sin(q1) + 1.68e-18*cos(q2)*sin(q1) + 1.0*cos(q4)*(1.68e-18*sin(q1) - 0.0274*cos(q1)*sin(q2) - 0.0274*cos(q2)*sin(q1) + 1.0*sin(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) + 1.0*cos(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1))) + 0.0274*sin(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) + 0.0274*cos(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1)) + 1.0*sin(q4)*(1.0*cos(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) - 1.0*sin(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1)))) - 1.0*sin(q5)*(sin(q4)*(1.68e-18*sin(q1) - 0.0274*cos(q1)*sin(q2) - 0.0274*cos(q2)*sin(q1) + 1.0*sin(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) + 1.0*cos(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1))) - 1.0*cos(q4)*(1.0*cos(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) - 1.0*sin(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1)))) - 6.29e-51*sin(q1) + 1.03e-34*cos(q1)*sin(q2) + 1.03e-34*cos(q2)*sin(q1) - 0.0274*cos(q4)*(1.68e-18*sin(q1) - 0.0274*cos(q1)*sin(q2) - 0.0274*cos(q2)*sin(q1) + 1.0*sin(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) + 1.0*cos(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1))) + 1.68e-18*sin(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) + 1.68e-18*cos(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1)) - 0.0274*sin(q4)*(1.0*cos(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) - 1.0*sin(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1)))) - 6.55e-19*sin(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) - 6.55e-19*cos(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1)) - 1.0*c*(0.0274*sin(q5)*(sin(q4)*(1.68e-18*sin(q1) - 0.0274*cos(q1)*sin(q2) - 0.0274*cos(q2)*sin(q1) + 1.0*sin(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) + 1.0*cos(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1))) - 1.0*cos(q4)*(1.0*cos(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) - 1.0*sin(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1)))) - 3.85e-67*sin(q1) - 0.0274*cos(q5)*(1.68e-18*cos(q1)*sin(q2) - 1.03e-34*sin(q1) + 1.68e-18*cos(q2)*sin(q1) + 1.0*cos(q4)*(1.68e-18*sin(q1) - 0.0274*cos(q1)*sin(q2) - 0.0274*cos(q2)*sin(q1) + 1.0*sin(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) + 1.0*cos(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1))) + 0.0274*sin(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) + 0.0274*cos(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1)) + 1.0*sin(q4)*(1.0*cos(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) - 1.0*sin(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1)))) + 6.29e-51*cos(q1)*sin(q2) + 6.29e-51*cos(q2)*sin(q1) - 1.68e-18*cos(q4)*(1.68e-18*sin(q1) - 0.0274*cos(q1)*sin(q2) - 0.0274*cos(q2)*sin(q1) + 1.0*sin(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) + 1.0*cos(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1))) + 1.03e-34*sin(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) + 1.03e-34*cos(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1)) - 1.68e-18*sin(q4)*(1.0*cos(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) - 1.0*sin(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1)))) + 0.0107*sin(q4)*(1.0*cos(q3)*(cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) - 1.0*sin(q3)*(0.0274*sin(q1) + 1.0*cos(q1)*sin(q2) + 0.999*cos(q2)*sin(q1)));
	j(1,1) = 0.0107*sin(q4)*(1.0*cos(q3)*(1.0*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) - 1.0*sin(q3)*(0.999*cos(q1)*sin(q2) + 1.0*cos(q2)*sin(q1))) - 0.011*cos(q1)*sin(q2) - 0.011*cos(q2)*sin(q1) - 1.0*b*(1.03e-34*cos(q1)*sin(q2) + 1.03e-34*cos(q2)*sin(q1) + 1.0*sin(q5)*(sin(q4)*(0.0274*cos(q1)*sin(q2) + 0.0274*cos(q2)*sin(q1) - 1.0*sin(q3)*(1.0*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) - 1.0*cos(q3)*(0.999*cos(q1)*sin(q2) + 1.0*cos(q2)*sin(q1))) + cos(q4)*(1.0*cos(q3)*(1.0*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) - 1.0*sin(q3)*(0.999*cos(q1)*sin(q2) + 1.0*cos(q2)*sin(q1)))) + 1.0*cos(q5)*(1.68e-18*cos(q1)*sin(q2) + 1.68e-18*cos(q2)*sin(q1) + 1.0*sin(q4)*(1.0*cos(q3)*(1.0*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) - 1.0*sin(q3)*(0.999*cos(q1)*sin(q2) + 1.0*cos(q2)*sin(q1))) + 0.0274*sin(q3)*(1.0*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) + 0.0274*cos(q3)*(0.999*cos(q1)*sin(q2) + 1.0*cos(q2)*sin(q1)) - 1.0*cos(q4)*(0.0274*cos(q1)*sin(q2) + 0.0274*cos(q2)*sin(q1) - 1.0*sin(q3)*(1.0*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) - 1.0*cos(q3)*(0.999*cos(q1)*sin(q2) + 1.0*cos(q2)*sin(q1)))) - 0.0274*sin(q4)*(1.0*cos(q3)*(1.0*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) - 1.0*sin(q3)*(0.999*cos(q1)*sin(q2) + 1.0*cos(q2)*sin(q1))) + 1.68e-18*sin(q3)*(1.0*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) + 1.68e-18*cos(q3)*(0.999*cos(q1)*sin(q2) + 1.0*cos(q2)*sin(q1)) + 0.0274*cos(q4)*(0.0274*cos(q1)*sin(q2) + 0.0274*cos(q2)*sin(q1) - 1.0*sin(q3)*(1.0*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) - 1.0*cos(q3)*(0.999*cos(q1)*sin(q2) + 1.0*cos(q2)*sin(q1)))) - 6.55e-19*sin(q3)*(1.0*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) - 1.0*c*(6.29e-51*cos(q1)*sin(q2) + 6.29e-51*cos(q2)*sin(q1) - 0.0274*sin(q5)*(sin(q4)*(0.0274*cos(q1)*sin(q2) + 0.0274*cos(q2)*sin(q1) - 1.0*sin(q3)*(1.0*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) - 1.0*cos(q3)*(0.999*cos(q1)*sin(q2) + 1.0*cos(q2)*sin(q1))) + cos(q4)*(1.0*cos(q3)*(1.0*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) - 1.0*sin(q3)*(0.999*cos(q1)*sin(q2) + 1.0*cos(q2)*sin(q1)))) - 0.0274*cos(q5)*(1.68e-18*cos(q1)*sin(q2) + 1.68e-18*cos(q2)*sin(q1) + 1.0*sin(q4)*(1.0*cos(q3)*(1.0*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) - 1.0*sin(q3)*(0.999*cos(q1)*sin(q2) + 1.0*cos(q2)*sin(q1))) + 0.0274*sin(q3)*(1.0*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) + 0.0274*cos(q3)*(0.999*cos(q1)*sin(q2) + 1.0*cos(q2)*sin(q1)) - 1.0*cos(q4)*(0.0274*cos(q1)*sin(q2) + 0.0274*cos(q2)*sin(q1) - 1.0*sin(q3)*(1.0*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) - 1.0*cos(q3)*(0.999*cos(q1)*sin(q2) + 1.0*cos(q2)*sin(q1)))) - 1.68e-18*sin(q4)*(1.0*cos(q3)*(1.0*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) - 1.0*sin(q3)*(0.999*cos(q1)*sin(q2) + 1.0*cos(q2)*sin(q1))) + 1.03e-34*sin(q3)*(1.0*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) + 1.03e-34*cos(q3)*(0.999*cos(q1)*sin(q2) + 1.0*cos(q2)*sin(q1)) + 1.68e-18*cos(q4)*(0.0274*cos(q1)*sin(q2) + 0.0274*cos(q2)*sin(q1) - 1.0*sin(q3)*(1.0*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) - 1.0*cos(q3)*(0.999*cos(q1)*sin(q2) + 1.0*cos(q2)*sin(q1)))) - 6.55e-19*cos(q3)*(0.999*cos(q1)*sin(q2) + 1.0*cos(q2)*sin(q1)) + 1.0*a*(1.0*cos(q5)*(sin(q4)*(0.0274*cos(q1)*sin(q2) + 0.0274*cos(q2)*sin(q1) - 1.0*sin(q3)*(1.0*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) - 1.0*cos(q3)*(0.999*cos(q1)*sin(q2) + 1.0*cos(q2)*sin(q1))) + cos(q4)*(1.0*cos(q3)*(1.0*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) - 1.0*sin(q3)*(0.999*cos(q1)*sin(q2) + 1.0*cos(q2)*sin(q1)))) - sin(q5)*(1.68e-18*cos(q1)*sin(q2) + 1.68e-18*cos(q2)*sin(q1) + 1.0*sin(q4)*(1.0*cos(q3)*(1.0*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) - 1.0*sin(q3)*(0.999*cos(q1)*sin(q2) + 1.0*cos(q2)*sin(q1))) + 0.0274*sin(q3)*(1.0*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) + 0.0274*cos(q3)*(0.999*cos(q1)*sin(q2) + 1.0*cos(q2)*sin(q1)) - 1.0*cos(q4)*(0.0274*cos(q1)*sin(q2) + 0.0274*cos(q2)*sin(q1) - 1.0*sin(q3)*(1.0*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) - 1.0*cos(q3)*(0.999*cos(q1)*sin(q2) + 1.0*cos(q2)*sin(q1))))) - 0.0107*cos(q4)*(0.0274*cos(q1)*sin(q2) + 0.0274*cos(q2)*sin(q1) - 1.0*sin(q3)*(1.0*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) - 1.0*cos(q3)*(0.999*cos(q1)*sin(q2) + 1.0*cos(q2)*sin(q1)));
	j(1,2) = 0.0107*cos(q4)*(1.0*cos(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) + 1.0*sin(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2))) - 6.55e-19*cos(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) - 1.0*b*(1.68e-18*cos(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) - 1.0*sin(q5)*(sin(q4)*(1.0*cos(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) + 1.0*sin(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2))) + 1.0*cos(q4)*(1.0*sin(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) - 1.0*cos(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)))) - 0.0274*cos(q4)*(1.0*cos(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) + 1.0*sin(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2))) + 1.0*cos(q5)*(0.0274*cos(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) + 1.0*cos(q4)*(1.0*cos(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) + 1.0*sin(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2))) + 0.0274*sin(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) - 1.0*sin(q4)*(1.0*sin(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) - 1.0*cos(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)))) + 1.68e-18*sin(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) + 0.0274*sin(q4)*(1.0*sin(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) - 1.0*cos(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)))) - 6.55e-19*sin(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) - 1.0*a*(cos(q5)*(sin(q4)*(1.0*cos(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) + 1.0*sin(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2))) + 1.0*cos(q4)*(1.0*sin(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) - 1.0*cos(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)))) + sin(q5)*(0.0274*cos(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) + 1.0*cos(q4)*(1.0*cos(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) + 1.0*sin(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2))) + 0.0274*sin(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) - 1.0*sin(q4)*(1.0*sin(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) - 1.0*cos(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2))))) - 1.0*c*(0.0274*sin(q5)*(sin(q4)*(1.0*cos(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) + 1.0*sin(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2))) + 1.0*cos(q4)*(1.0*sin(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) - 1.0*cos(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)))) + 1.03e-34*cos(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) - 1.68e-18*cos(q4)*(1.0*cos(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) + 1.0*sin(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2))) - 0.0274*cos(q5)*(0.0274*cos(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) + 1.0*cos(q4)*(1.0*cos(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) + 1.0*sin(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2))) + 0.0274*sin(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) - 1.0*sin(q4)*(1.0*sin(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) - 1.0*cos(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)))) + 1.03e-34*sin(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)) + 1.68e-18*sin(q4)*(1.0*sin(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) - 1.0*cos(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)))) - 0.0107*sin(q4)*(1.0*sin(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) - 1.0*cos(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)));
	j(1,3) = 0.0107*sin(q4)*(1.68e-18*cos(q1) - 0.0274*cos(q1)*cos(q2) + 0.0274*sin(q1)*sin(q2) - 1.0*sin(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) + 1.0*cos(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2))) + 0.0107*cos(q4)*(cos(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) + sin(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2))) - 1.0*a*(sin(q5)*(1.0*sin(q4)*(1.68e-18*cos(q1) - 0.0274*cos(q1)*cos(q2) + 0.0274*sin(q1)*sin(q2) - 1.0*sin(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) + 1.0*cos(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2))) + 1.0*cos(q4)*(cos(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) + sin(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)))) - 1.0*cos(q5)*(cos(q4)*(1.68e-18*cos(q1) - 0.0274*cos(q1)*cos(q2) + 0.0274*sin(q1)*sin(q2) - 1.0*sin(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) + 1.0*cos(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2))) - 1.0*sin(q4)*(cos(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) + sin(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2))))) + c*(0.0274*cos(q5)*(1.0*sin(q4)*(1.68e-18*cos(q1) - 0.0274*cos(q1)*cos(q2) + 0.0274*sin(q1)*sin(q2) - 1.0*sin(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) + 1.0*cos(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2))) + 1.0*cos(q4)*(cos(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) + sin(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)))) + 1.68e-18*sin(q4)*(1.68e-18*cos(q1) - 0.0274*cos(q1)*cos(q2) + 0.0274*sin(q1)*sin(q2) - 1.0*sin(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) + 1.0*cos(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2))) + 1.68e-18*cos(q4)*(cos(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) + sin(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2))) + 0.0274*sin(q5)*(cos(q4)*(1.68e-18*cos(q1) - 0.0274*cos(q1)*cos(q2) + 0.0274*sin(q1)*sin(q2) - 1.0*sin(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) + 1.0*cos(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2))) - 1.0*sin(q4)*(cos(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) + sin(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2))))) - 1.0*b*(1.0*cos(q5)*(1.0*sin(q4)*(1.68e-18*cos(q1) - 0.0274*cos(q1)*cos(q2) + 0.0274*sin(q1)*sin(q2) - 1.0*sin(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) + 1.0*cos(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2))) + 1.0*cos(q4)*(cos(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) + sin(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)))) - 0.0274*sin(q4)*(1.68e-18*cos(q1) - 0.0274*cos(q1)*cos(q2) + 0.0274*sin(q1)*sin(q2) - 1.0*sin(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) + 1.0*cos(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2))) - 0.0274*cos(q4)*(cos(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) + sin(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2))) + 1.0*sin(q5)*(cos(q4)*(1.68e-18*cos(q1) - 0.0274*cos(q1)*cos(q2) + 0.0274*sin(q1)*sin(q2) - 1.0*sin(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) + 1.0*cos(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2))) - 1.0*sin(q4)*(cos(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) + sin(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)))));
	j(1,4) = 1.0*c*(0.0274*cos(q5)*(sin(q4)*(1.68e-18*cos(q1) - 0.0274*cos(q1)*cos(q2) + 0.0274*sin(q1)*sin(q2) - 1.0*sin(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) + 1.0*cos(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2))) + cos(q4)*(cos(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) + sin(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)))) - 0.0274*sin(q5)*(1.03e-34*cos(q1) - 1.68e-18*cos(q1)*cos(q2) - 1.0*cos(q4)*(1.68e-18*cos(q1) - 0.0274*cos(q1)*cos(q2) + 0.0274*sin(q1)*sin(q2) - 1.0*sin(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) + 1.0*cos(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2))) + 1.68e-18*sin(q1)*sin(q2) + 0.0274*sin(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) + 1.0*sin(q4)*(cos(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) + sin(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2))) - 0.0274*cos(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)))) - 1.0*a*(sin(q5)*(sin(q4)*(1.68e-18*cos(q1) - 0.0274*cos(q1)*cos(q2) + 0.0274*sin(q1)*sin(q2) - 1.0*sin(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) + 1.0*cos(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2))) + cos(q4)*(cos(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) + sin(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)))) + cos(q5)*(1.03e-34*cos(q1) - 1.68e-18*cos(q1)*cos(q2) - 1.0*cos(q4)*(1.68e-18*cos(q1) - 0.0274*cos(q1)*cos(q2) + 0.0274*sin(q1)*sin(q2) - 1.0*sin(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) + 1.0*cos(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2))) + 1.68e-18*sin(q1)*sin(q2) + 0.0274*sin(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) + 1.0*sin(q4)*(cos(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) + sin(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2))) - 0.0274*cos(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)))) - 1.0*b*(1.0*cos(q5)*(sin(q4)*(1.68e-18*cos(q1) - 0.0274*cos(q1)*cos(q2) + 0.0274*sin(q1)*sin(q2) - 1.0*sin(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) + 1.0*cos(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2))) + cos(q4)*(cos(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) + sin(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2)))) - 1.0*sin(q5)*(1.03e-34*cos(q1) - 1.68e-18*cos(q1)*cos(q2) - 1.0*cos(q4)*(1.68e-18*cos(q1) - 0.0274*cos(q1)*cos(q2) + 0.0274*sin(q1)*sin(q2) - 1.0*sin(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) + 1.0*cos(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2))) + 1.68e-18*sin(q1)*sin(q2) + 0.0274*sin(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) + 1.0*sin(q4)*(cos(q3)*(1.0*cos(q1)*sin(q2) + cos(q2)*sin(q1)) + sin(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2))) - 0.0274*cos(q3)*(0.0274*cos(q1) + 0.999*cos(q1)*cos(q2) - 1.0*sin(q1)*sin(q2))));

	j(2,1) = 1.0*c*(0.0274*cos(q5)*(1.68e-18*sin(q2) + 0.0274*cos(q2)*sin(q3) + 0.0274*cos(q3)*sin(q2) + 1.0*sin(q4)*(cos(q2)*cos(q3) - 1.0*sin(q2)*sin(q3)) + 1.0*cos(q4)*(1.0*cos(q2)*sin(q3) - 0.0274*sin(q2) + 0.999*cos(q3)*sin(q2))) - 1.03e-34*cos(q2)*sin(q3) - 1.03e-34*cos(q3)*sin(q2) - 6.29e-51*sin(q2) + 1.68e-18*sin(q4)*(cos(q2)*cos(q3) - 1.0*sin(q2)*sin(q3)) + 1.68e-18*cos(q4)*(1.0*cos(q2)*sin(q3) - 0.0274*sin(q2) + 0.999*cos(q3)*sin(q2)) + 0.0274*sin(q5)*(1.0*cos(q4)*(cos(q2)*cos(q3) - 1.0*sin(q2)*sin(q3)) - 1.0*sin(q4)*(1.0*cos(q2)*sin(q3) - 0.0274*sin(q2) + 0.999*cos(q3)*sin(q2)))) - 6.55e-19*cos(q2)*sin(q3) - 6.54e-19*cos(q3)*sin(q2) - 0.011*sin(q2) + 0.0107*sin(q4)*(cos(q2)*cos(q3) - 1.0*sin(q2)*sin(q3)) + 0.0107*cos(q4)*(1.0*cos(q2)*sin(q3) - 0.0274*sin(q2) + 0.999*cos(q3)*sin(q2)) - 1.0*a*(sin(q5)*(1.68e-18*sin(q2) + 0.0274*cos(q2)*sin(q3) + 0.0274*cos(q3)*sin(q2) + 1.0*sin(q4)*(cos(q2)*cos(q3) - 1.0*sin(q2)*sin(q3)) + 1.0*cos(q4)*(1.0*cos(q2)*sin(q3) - 0.0274*sin(q2) + 0.999*cos(q3)*sin(q2))) - 1.0*cos(q5)*(1.0*cos(q4)*(cos(q2)*cos(q3) - 1.0*sin(q2)*sin(q3)) - 1.0*sin(q4)*(1.0*cos(q2)*sin(q3) - 0.0274*sin(q2) + 0.999*cos(q3)*sin(q2)))) - 1.0*b*(1.03e-34*sin(q2) + 1.68e-18*cos(q2)*sin(q3) + 1.68e-18*cos(q3)*sin(q2) + 1.0*cos(q5)*(1.68e-18*sin(q2) + 0.0274*cos(q2)*sin(q3) + 0.0274*cos(q3)*sin(q2) + 1.0*sin(q4)*(cos(q2)*cos(q3) - 1.0*sin(q2)*sin(q3)) + 1.0*cos(q4)*(1.0*cos(q2)*sin(q3) - 0.0274*sin(q2) + 0.999*cos(q3)*sin(q2))) - 0.0274*sin(q4)*(cos(q2)*cos(q3) - 1.0*sin(q2)*sin(q3)) - 0.0274*cos(q4)*(1.0*cos(q2)*sin(q3) - 0.0274*sin(q2) + 0.999*cos(q3)*sin(q2)) + 1.0*sin(q5)*(1.0*cos(q4)*(cos(q2)*cos(q3) - 1.0*sin(q2)*sin(q3)) - 1.0*sin(q4)*(1.0*cos(q2)*sin(q3) - 0.0274*sin(q2) + 0.999*cos(q3)*sin(q2))));
	j(2,2) = 0.0107*cos(q4)*(1.0*cos(q3)*sin(q2) + 1.0*sin(q3)*(1.0*cos(q2) - 6.12e-17)) - 1.0*b*(1.0*cos(q5)*(0.0274*cos(q3)*sin(q2) + 1.0*cos(q4)*(1.0*cos(q3)*sin(q2) + 1.0*sin(q3)*(1.0*cos(q2) - 6.12e-17)) - 1.0*sin(q4)*(sin(q2)*sin(q3) - 1.0*cos(q3)*(1.0*cos(q2) - 6.12e-17)) + 0.0274*sin(q3)*(1.0*cos(q2) - 6.12e-17)) + 1.68e-18*cos(q3)*sin(q2) - 0.0274*cos(q4)*(1.0*cos(q3)*sin(q2) + 1.0*sin(q3)*(1.0*cos(q2) - 6.12e-17)) - 1.0*sin(q5)*(sin(q4)*(1.0*cos(q3)*sin(q2) + 1.0*sin(q3)*(1.0*cos(q2) - 6.12e-17)) + cos(q4)*(sin(q2)*sin(q3) - 1.0*cos(q3)*(1.0*cos(q2) - 6.12e-17))) + 0.0274*sin(q4)*(sin(q2)*sin(q3) - 1.0*cos(q3)*(1.0*cos(q2) - 6.12e-17)) + 1.68e-18*sin(q3)*(1.0*cos(q2) - 6.12e-17)) - 6.55e-19*cos(q3)*sin(q2) - 0.0107*sin(q4)*(sin(q2)*sin(q3) - 1.0*cos(q3)*(1.0*cos(q2) - 6.12e-17)) - 6.55e-19*sin(q3)*(1.0*cos(q2) - 6.12e-17) - 1.0*c*(1.03e-34*cos(q3)*sin(q2) - 0.0274*cos(q5)*(0.0274*cos(q3)*sin(q2) + 1.0*cos(q4)*(1.0*cos(q3)*sin(q2) + 1.0*sin(q3)*(1.0*cos(q2) - 6.12e-17)) - 1.0*sin(q4)*(sin(q2)*sin(q3) - 1.0*cos(q3)*(1.0*cos(q2) - 6.12e-17)) + 0.0274*sin(q3)*(1.0*cos(q2) - 6.12e-17)) - 1.68e-18*cos(q4)*(1.0*cos(q3)*sin(q2) + 1.0*sin(q3)*(1.0*cos(q2) - 6.12e-17)) + 0.0274*sin(q5)*(sin(q4)*(1.0*cos(q3)*sin(q2) + 1.0*sin(q3)*(1.0*cos(q2) - 6.12e-17)) + cos(q4)*(sin(q2)*sin(q3) - 1.0*cos(q3)*(1.0*cos(q2) - 6.12e-17))) + 1.68e-18*sin(q4)*(sin(q2)*sin(q3) - 1.0*cos(q3)*(1.0*cos(q2) - 6.12e-17)) + 1.03e-34*sin(q3)*(1.0*cos(q2) - 6.12e-17)) - 1.0*a*(sin(q5)*(0.0274*cos(q3)*sin(q2) + 1.0*cos(q4)*(1.0*cos(q3)*sin(q2) + 1.0*sin(q3)*(1.0*cos(q2) - 6.12e-17)) - 1.0*sin(q4)*(sin(q2)*sin(q3) - 1.0*cos(q3)*(1.0*cos(q2) - 6.12e-17)) + 0.0274*sin(q3)*(1.0*cos(q2) - 6.12e-17)) + cos(q5)*(sin(q4)*(1.0*cos(q3)*sin(q2) + 1.0*sin(q3)*(1.0*cos(q2) - 6.12e-17)) + cos(q4)*(sin(q2)*sin(q3) - 1.0*cos(q3)*(1.0*cos(q2) - 6.12e-17))));
	j(2,3) = 0.0107*cos(q4)*(cos(q3)*sin(q2) + sin(q3)*(1.0*cos(q2) - 6.12e-17)) - 0.0107*sin(q4)*(0.0274*cos(q2) + 1.0*sin(q2)*sin(q3) - 1.0*cos(q3)*(1.0*cos(q2) - 6.12e-17) + 3.75e-33) - 1.0*a*(1.0*cos(q5)*(cos(q4)*(0.0274*cos(q2) + 1.0*sin(q2)*sin(q3) - 1.0*cos(q3)*(1.0*cos(q2) - 6.12e-17) + 3.75e-33) + sin(q4)*(cos(q3)*sin(q2) + sin(q3)*(1.0*cos(q2) - 6.12e-17))) + sin(q5)*(1.0*cos(q4)*(cos(q3)*sin(q2) + sin(q3)*(1.0*cos(q2) - 6.12e-17)) - 1.0*sin(q4)*(0.0274*cos(q2) + 1.0*sin(q2)*sin(q3) - 1.0*cos(q3)*(1.0*cos(q2) - 6.12e-17) + 3.75e-33))) + 1.0*c*(0.0274*cos(q5)*(1.0*cos(q4)*(cos(q3)*sin(q2) + sin(q3)*(1.0*cos(q2) - 6.12e-17)) - 1.0*sin(q4)*(0.0274*cos(q2) + 1.0*sin(q2)*sin(q3) - 1.0*cos(q3)*(1.0*cos(q2) - 6.12e-17) + 3.75e-33)) - 0.0274*sin(q5)*(cos(q4)*(0.0274*cos(q2) + 1.0*sin(q2)*sin(q3) - 1.0*cos(q3)*(1.0*cos(q2) - 6.12e-17) + 3.75e-33) + sin(q4)*(cos(q3)*sin(q2) + sin(q3)*(1.0*cos(q2) - 6.12e-17))) + 1.68e-18*cos(q4)*(cos(q3)*sin(q2) + sin(q3)*(1.0*cos(q2) - 6.12e-17)) - 1.68e-18*sin(q4)*(0.0274*cos(q2) + 1.0*sin(q2)*sin(q3) - 1.0*cos(q3)*(1.0*cos(q2) - 6.12e-17) + 3.75e-33)) - 1.0*b*(1.0*cos(q5)*(1.0*cos(q4)*(cos(q3)*sin(q2) + sin(q3)*(1.0*cos(q2) - 6.12e-17)) - 1.0*sin(q4)*(0.0274*cos(q2) + 1.0*sin(q2)*sin(q3) - 1.0*cos(q3)*(1.0*cos(q2) - 6.12e-17) + 3.75e-33)) - 1.0*sin(q5)*(cos(q4)*(0.0274*cos(q2) + 1.0*sin(q2)*sin(q3) - 1.0*cos(q3)*(1.0*cos(q2) - 6.12e-17) + 3.75e-33) + sin(q4)*(cos(q3)*sin(q2) + sin(q3)*(1.0*cos(q2) - 6.12e-17))) - 0.0274*cos(q4)*(cos(q3)*sin(q2) + sin(q3)*(1.0*cos(q2) - 6.12e-17)) + 0.0274*sin(q4)*(0.0274*cos(q2) + 1.0*sin(q2)*sin(q3) - 1.0*cos(q3)*(1.0*cos(q2) - 6.12e-17) + 3.75e-33));
	j(2,4) = 1.0*c*(0.0274*cos(q5)*(1.0*cos(q4)*(cos(q3)*sin(q2) + sin(q3)*(1.0*cos(q2) - 6.12e-17)) - 1.0*sin(q4)*(0.0274*cos(q2) + 1.0*sin(q2)*sin(q3) - 1.0*cos(q3)*(1.0*cos(q2) - 6.12e-17) + 3.75e-33)) + 0.0274*sin(q5)*(1.68e-18*cos(q2) - 0.0274*sin(q2)*sin(q3) - 1.0*cos(q4)*(0.0274*cos(q2) + 1.0*sin(q2)*sin(q3) - 1.0*cos(q3)*(1.0*cos(q2) - 6.12e-17) + 3.75e-33) - 1.0*sin(q4)*(cos(q3)*sin(q2) + sin(q3)*(1.0*cos(q2) - 6.12e-17)) + 0.0274*cos(q3)*(1.0*cos(q2) - 6.12e-17) + 2.3e-49)) - 1.0*b*(1.0*cos(q5)*(1.0*cos(q4)*(cos(q3)*sin(q2) + sin(q3)*(1.0*cos(q2) - 6.12e-17)) - 1.0*sin(q4)*(0.0274*cos(q2) + 1.0*sin(q2)*sin(q3) - 1.0*cos(q3)*(1.0*cos(q2) - 6.12e-17) + 3.75e-33)) + 1.0*sin(q5)*(1.68e-18*cos(q2) - 0.0274*sin(q2)*sin(q3) - 1.0*cos(q4)*(0.0274*cos(q2) + 1.0*sin(q2)*sin(q3) - 1.0*cos(q3)*(1.0*cos(q2) - 6.12e-17) + 3.75e-33) - 1.0*sin(q4)*(cos(q3)*sin(q2) + sin(q3)*(1.0*cos(q2) - 6.12e-17)) + 0.0274*cos(q3)*(1.0*cos(q2) - 6.12e-17) + 2.3e-49)) - 1.0*a*(sin(q5)*(1.0*cos(q4)*(cos(q3)*sin(q2) + sin(q3)*(1.0*cos(q2) - 6.12e-17)) - 1.0*sin(q4)*(0.0274*cos(q2) + 1.0*sin(q2)*sin(q3) - 1.0*cos(q3)*(1.0*cos(q2) - 6.12e-17) + 3.75e-33)) - 1.0*cos(q5)*(1.68e-18*cos(q2) - 0.0274*sin(q2)*sin(q3) - 1.0*cos(q4)*(0.0274*cos(q2) + 1.0*sin(q2)*sin(q3) - 1.0*cos(q3)*(1.0*cos(q2) - 6.12e-17) + 3.75e-33) - 1.0*sin(q4)*(cos(q3)*sin(q2) + sin(q3)*(1.0*cos(q2) - 6.12e-17)) + 0.0274*cos(q3)*(1.0*cos(q2) - 6.12e-17) + 2.3e-49));

	//j = MatrixXd::Zero(3,7);	//comment if jacobian needed

	return j;
}

//these two matrices are too big for matlab to show..
MatrixXd J6(VectorXd& q) {
	MatrixXd j = MatrixXd::Zero(3,7); //3 cartesian dimensions, 7 joints
	double q1 = q[0], q2 = q[1], q3 = q[2], q4 = q[3], q5 = q[4], q6 = q[5], q7 = q[6];


	return j;
}

MatrixXd J7(VectorXd& q) {
	MatrixXd j = MatrixXd::Zero(3,7); //3 cartesian dimensions, 7 joints
	double q1 = q[0], q2 = q[1], q3 = q[2], q4 = q[3], q5 = q[4], q6 = q[5], q7 = q[6];


	return j;
}
//computes the Jacobian of the closest point to the obstacle, uses q and closestJoint and a,b,c (from subscriber)
MatrixXd getJ0(VectorXd& q) {
	//cout << "J0 q1: " << q[0] << endl;
	MatrixXd J0 = MatrixXd::Zero(3,7);
	switch(closestJoint) {
		case 1: J0 = J1(q);
				//cout << "J1" << endl;
			break;
		case 2:	J0 = J2(q);
				//cout << "J2" << endl;
			break;
		case 3: J0 = J3(q);
				//cout << "J3" << endl;
			break;
		case 4: J0 = J4(q);
				//cout << "J4" << endl;
			break;
		case 5: J0 = J5(q);
				//cout << "J5" << endl;
			break;
		case 6: J0 = J6(q);
				//cout << "J6" << endl;
			break;
		case 7: J0 = J7(q);
				//cout << "J7" << endl;
			break;
		default: cout << "Error computing J0." << endl;
			break;
	}
	return J0;
}


VectorXd getCurrentVelocities(MatrixXd& ys, MatrixXd& yds, VectorXd& prevAngles, VectorXd& ts, int t) {
	MatrixXd J(3,7), Jp(7,3), Jt(7,3), J0(3,7);	//all 7 joint angles for all times
	VectorXd q(7), term1(7), yds_block(3);
	VectorXd currentVelocities(7);	//to be published
	double timeStep = ts[1] - ts[0];

	q = prevAngles;

	J = getJacobian(q);
	J0 = getJ0(q);
	Jp = getPseudoInverse(J);  	//pseudoinverse is very unstable 
	Jt = J.transpose();				//transpose is not very precise, but still better, so it's being used in term1

	Vector3d deltaY = Vector3d::Zero();
	if(t < ts.size()-1) {
		deltaY = ys.block(t+1,0,1,3).transpose()-getEndEffectorPosition(q);	//the difference between the real end effector position and the actual position of the dmp
	}

	for(int i = 0; i<3; i++) {
		yds_block[i] = yds(t,i);		//velocity in the current time step 
	}

	//term1 = Jp*yds_block;	//inverse kinematics
	term1 = Jp*deltaY/timeStep;
	//null space constraints
	//x0d comes from vrep via subscriber (global variable)
	MatrixXd term2 = J0*(MatrixXd::Identity(7,7) - Jp*J);
	Vector3d test; test << 0,0,-100000;
	VectorXd term3 = getPseudoInverse(term2)*(x0d - J0*Jp*yds_block);
	currentVelocities =  term1 + 0.1*term3;		//add here the IK term

	for(int i = 0; i < 7; i++) {
		angleVelocities(t,i) = currentVelocities[i];	//we store all the angular velocities in a matrix for the cost computation
	}
	return currentVelocities;
}

void checkConstraints(VectorXd& angles) {
	//check if these angles fit the kuka constraints and fix them
	double const170 = 165*PI/180;	//+/-170 degrees is the limit for joint 0,2,4,6
	double const120 = 115*PI/180;	//+/-120 degrees is the limit for joint 1,3,5
	for(int i = 0; i < 7; i +=2) {
		if(angles[i] < -const170) {
			angles[i] = -const170;
		}
		if(angles[i] > const170) {
			angles[i] = const170;
		}
	}
	for(int i = 1; i < 7; i +=2) {
		if(angles[i] < -const120) {
			angles[i] = -const120;
		}
		if(angles[i] > const120) {
			angles[i] = const120;
		}
	}
}
void integrate(VectorXd& prevValue, VectorXd& derivative, double timeStep, MatrixXd& result, int t){
	if(t > 0) {	//t < result.rows()-1
		for(int i = 0; i<prevValue.size(); i++) {
			result(t,i) = prevValue[i] + 1.2*derivative[i]*timeStep;
		}
	}
}