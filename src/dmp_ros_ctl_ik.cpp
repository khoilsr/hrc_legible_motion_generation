// this program should publish messages to control the robot using dmp in combination with inverse kinematics 
#include <string>
#include <sstream>
#include <fstream> 
#include <set>
#include <vector>
#include <map>
#include <limits>
#include <cmath>
#include <math.h>
#include <cstdio>
#include <thread>
#include <chrono>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include "SDL.h"
#include <keyboard/Key.h>

#include "ros/ros.h"
#include "std_msgs/String.h"
#include "std_msgs/Float32.h"
#include "std_msgs/Bool.h"
#include "geometry_msgs/Point.h"
#include "geometry_msgs/PoseStamped.h"
#include "sensor_msgs/JointState.h"
#include "trajectory_msgs/JointTrajectory.h"
#include "trajectory_msgs/JointTrajectoryPoint.h"
#include <qualisys/Subject.h>
#include <std_msgs/Float64MultiArray.h>
#include <std_msgs/Float64.h>
#include <std_msgs/Int64.h>
#include <std_msgs/Int64MultiArray.h>
#include <std_msgs/Int32.h>

#include "DmpSolver.hpp"			//self-made class that creates a whole dmp solver
#include "Kinematics.hpp"


using namespace std;
using namespace Eigen;
using namespace DmpBbo;

const double PI = 3.141592653589793238463;

ros::Publisher vrepPub;	//sends qualisys data to vrep
ros::Publisher predictionPub; //sends ptomp data to vrep 

Kinematics kinematics;	//object of class kinematics, responsible for all kinematics calculations

//global variables, get filled by callback function from subscriber
Vector3d obstaclePosition = Vector3d::Zero(), obstacleVelocity = Vector3d::Zero(), robotPosition = Vector3d::Zero();


int closestJoint = 2;			//number of joint closest to obstacle (from 1 to 7) 
bool collision = false;			//gets used in computeCosts
double a = 0.5, b = 0.03, c = 1;					//values in vector [a, b, c, 1], get multiplied by forward transformation to compute J0 (see Hoffmann paper)
Vector3d x0d = Vector3d::Zero();					//velocity of the closest point to the obstacle, needed for null space contraint
Vector3d handPosition = Vector3d::Zero(), handVelocity = Vector3d::Zero();
Vector3d robotBasePosition = Vector3d::Zero();
Vector3d futurePos1 = Vector3d::Zero();
Vector3d futurePos2 = Vector3d::Zero();

double humanDuration = 0, humanTime = 3;
int humanGoal = 0;

double startOfRobotMovement; 	//contains the time at the start of the movement of the robot, is used to compute the human reaction time in getHumanTime()
MatrixXd distances; 			//contains the distance between robot and human for all trajectories over time (ts.size()xnUpdates*nSamplesPerUpdate) 

//variables to control real robot - kukabot
VectorXd curKukabotPos = VectorXd::Zero(7);			//current position
VectorXd curKukabotVel = VectorXd::Zero(7);			//current velocity
VectorXd curKukabotAcc = VectorXd::Zero(7);			//current acceleration

//global ros messages for publisher (for impedance controller)
std_msgs::Float64MultiArray  msg_positions, msg_torque, msg_gains;

//function to publish trajectory messages to the robot, implemented below main()
//void publishMessages(VectorXd& angles, VectorXd& currentAngleVelocities, ros::Publisher& trajPub);

//publishes the same messages but in a different format for the impedance controller
void publishMessagesImpedance(VectorXd& angles, VectorXd& angleVelo, ros::Publisher& pub_positions, ros::Publisher& pub_torque, ros::Publisher& pub_gains);

//integtrates a Matrix, with a specified startValue and a time steps, writes the result in result. was tested, it gives similar results to the computed values \o/
VectorXd integrateVector(VectorXd& prevValue, VectorXd& derivative, double timeStep);
//function takes costVariables and computes cost, implemented below
void computeCosts(MatrixXd& ys, MatrixXd& yds, MatrixXd& ydds, int nDims, VectorXd& cost, VectorXd& ts, bool collisionOccured, MatrixXd& angles, MatrixXd& angleVelocities, MatrixXd& ydReal, double robot_time, bool accuracy,
					MatrixXd& y1, MatrixXd& y2);

//ckecks if the computed angles are within the kuka con
//void checkConstraints(VectorXd& angles);

//saves trajectory and cost data to file 
void writeToFile(int subjectNr, int experimentNr, int updateNr, int goalNr, int sampleNr, VectorXd& data);

//saves all published angles in a file
void writeToFileAngles(int subjectNr, int experimentNr, int updateNr, int goalNr, int sampleNr, MatrixXd& data);

//saves all dmp calculated trajectories (before inverse kinematics to compare)
void writeToFileDmpEndEff(int subjectNr, int experimentNr, int updateNr, int goalNr, int sampleNr, MatrixXd& data);

//saves all data required to load a trajectory: ts, y, yd, ydd
void writeToFileTrajectory(int subjectNr, int experimentNr, int updateNr, int goalNr, int sampleNr, VectorXd& ts, MatrixXd& y, MatrixXd& yd, MatrixXd& ydd);

//saves the weights for the gaussian basis functions after each update
void writeToFileWeights(int subjectNr, int experimentNr, int updateNr, int goalNr, VectorXd& data);

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
	double factor;

	if (dist==0)
		factor = 0;
	else
		factor = 1/dist;

	return factor*pos;
}

// function to create a repulsive push in the joint that close to the limit 
void jointlimitForce(double& currentJointAng, double& currentJointVel, double desiredJointAng, double desiredJointVel, double jointlimit, double timeStep)
{
	double Fmax = 5;
	double thres_col = 5*PI/180;   //5 degrees
	double alpha = 4; //shape factor
    double dist = abs(currentJointAng - jointlimit);


	double force_val = Fmax/(1+exp((dist*(2/thres_col)-1)*alpha));

	if ((currentJointAng - jointlimit) <0)
		force_val = -force_val;

	if(abs(currentJointAng - jointlimit) > thres_col) {
		force_val = 0;
	}

	double e = currentJointAng - desiredJointAng;		//error position p - pdesired
	double e_dot = currentJointVel - desiredJointVel;	//error in velocity v - vdesired

	double M = 1;
	double K = 150;						//100
	double D = 1.5*sqrt(4*K)*M;      // 1.5 times of critical damping for smooth motion

	double e_dotdot = (1/M)*(force_val - D*e_dot - K*e);	//gets calculated to minimize e

	currentJointVel = currentJointVel + e_dotdot*timeStep;
	currentJointAng = currentJointVel + e_dot*timeStep;


}

//function to compute the repellent force resulting from the dynamic potential field for the end effector
//it needs xds: velocity of end effector, xs: position of end effector, x_obs: position of obstacle, xd_obs: velocity of obstacle
//VectorXd dynamicPotentialForce(VectorXd& currentY, MatrixXd& yds, int nDims, VectorXd& ts, int t, int counter) {
VectorXd dynamicPotentialForce(VectorXd& currentY, VectorXd& currentV) {
	double lambda = 50, beta = 2;  					//parameters to include in policy improvement
	VectorXd force = VectorXd::Zero(3);
	// Vector3d ypos = currentY;		//better to use here the closest point to obstacle, now it's the position of the end effector
	Vector3d ypos = robotPosition;
	Vector3d vel = currentV; 

	Vector3d x_obs, xd_obs; 		//position and velocity of the obstacle

	x_obs = obstaclePosition; 
	xd_obs = obstacleVelocity;								//obstacle position and velocity from vrep, change to handPosition and handVelocity when working with qualisys
	Vector3d pos = ypos - x_obs;	//relative position to the obstacle
	double dist = pos.norm();		//distance between obstacle and end effector p(x)

	// if(dist < 0.04) {
	// 	cout << "too near to end effector!!" << endl;
	// 	cout << "dist " << dist << endl;
	// 	cout << "ypos " << endl << ypos << endl;
	// 	cout << "obs pos " << endl << x_obs << endl;
	// }
	//distances(t, counter) = dist;	//save in a big matrix for storing in a file

	Vector3d rel_vel = vel - xd_obs;	//relative velocity
	//the following calculation was adapted from the dynamic potential field equations presented in Hoffmann's paper
	double temp = rel_vel.norm()*dist;
	double cos_theta = (double)(rel_vel.transpose()*pos)/temp;	

	Vector3d temp2 = rel_vel.cross(pos);
	double theta = atan2(temp2.norm(), rel_vel.dot(pos));	//only from 0 to pi
	cos_theta = cos(theta);
	//force = lambda * pow(-cos_theta, beta-1) * rel_vel.norm()/dist * (beta*getGradientCos(vel, pos, dist) - cos_theta/dist*getGradientDistance(dist, pos));
	//cout << "theta " << theta << endl;
	//if(acos(cos_theta) < PI/2 && acos(cos_theta) > 0) {
	if(theta < PI/2 && theta > 0 && dist < 0.2) {
		//cout << "dist= " << dist << endl;
		force = lambda * pow(-cos_theta, beta-1) * rel_vel.norm()/dist * (beta*getGradientCos(vel, pos, dist) - cos_theta/dist*getGradientDistance(dist, pos));	
		//cout << "force was applied. right" << endl;
	}	
	//else  if (acos(cos_theta) > -PI/2 && acos(cos_theta) < 0) {
	else  if (theta > -PI/2 && theta < 0 && dist < 0.2) {				//never gets called (from experimental observation)
		//cos_theta = - cos_theta;
		//cout << "2dist= " << dist << endl;
		force = lambda * pow(-cos_theta, beta-1) * rel_vel.norm()/dist * (beta*getGradientCos(vel, pos, dist) - cos_theta/dist*getGradientDistance(dist, pos));
		//force = Vector3d::Zero();
		//cout << "force was applied. left" << endl;
	}
	else {
		//force = Vector3d::Zero();
	}


	// This is a different potential field approach with fixed threshold and force/////////////////////////////
	double Fmax = 200; //force maximum
	double alpha = 4; //shape factor 
	double thres_col = 0.30;
	double force_val;

	force_val = Fmax/(1+exp((dist*(2/thres_col)-1)*alpha));
	if(dist > thres_col) {
		force_val = 0;
	}


	//std::cout << "Potential force: " << force_val << std::endl;
	//Vector3d base_pos(-0.536, 2.678, x_obs(2));   // same height as obstacle --currently hardcoded for testing
	Vector3d force_vec = (ypos-x_obs);
	double temp_value = force_vec(1);
	force_vec(1) = -force_vec(0);
	force_vec(0) = temp_value;
	// if (force_vec(1)>0) {
	// 	force_vec(1) = 0;    // prevent the robot to go further on y direction
	// }

	Vector3d force_future_vec_1 = (ypos-futurePos1);
	temp_value = force_future_vec_1(1);
	force_future_vec_1(1) = -force_future_vec_1(0);
	force_future_vec_1(0) = temp_value;
	// if (force_future_vec_1(1)>0) {
	// 	force_future_vec_1(1) = 0;    // prevent the robot to go further on y direction
	// }

	Vector3d force_future_vec_2 = (ypos-futurePos2);
	temp_value = force_future_vec_2(1);
	force_future_vec_2(1) = -force_future_vec_2(0);
	force_future_vec_2(0) = temp_value;
	// if (force_future_vec_2(1)>0) {
	// 	force_future_vec_2(1) = 0;    // prevent the robot to go further on y direction
	// }

	double force_fur_val1 = 0;
	double force_fur_val2 = 0;
	VectorXd force_fur1 = VectorXd::Zero(3);
	VectorXd force_fur2 = VectorXd::Zero(3);

//	cout << "Human velocity: " << xd_obs.norm() << endl;
//	cout << "Human velocity vector: " << xd_obs(0) << "' " << xd_obs(1) << ", " << xd_obs(2) << endl;

	// if (futurePos1.norm()>0)
	// {
	// 	cout << "Human velocity: " << xd_obs.norm() << endl;
	// 	cout << "Human velocity vector: " << xd_obs(0) << "' " << xd_obs(1) << ", " << xd_obs(2) << endl;
	// }

	force = force_val*getGradientDistance(force_vec.norm(),force_vec);

	if (futurePos1.norm()>0)	  //  future prediction is added if the human move and move toward the goal
	{

		double fur_dist1 = force_future_vec_1.norm();
		double fur_dist2 = force_future_vec_2.norm();
		force_fur_val1 = Fmax/(1+exp((fur_dist1*(2/thres_col)-1)*alpha));
//		if(fur_dist1 > thres_col) {	force_fur_val1 = 0;	}
		force_fur_val2 = Fmax/(1+exp((fur_dist2*(2/thres_col)-1)*alpha));
//		if(fur_dist2 > thres_col) {	force_fur_val2 = 0;	}


	}		

	
	force_fur1 = force_fur_val1*getGradientDistance(force_future_vec_1.norm(),force_future_vec_1);
	force_fur2 = force_fur_val2*getGradientDistance(force_future_vec_2.norm(),force_future_vec_2);
	// if (futurePos1.norm()>0)
	// {
	// 	cout << "force: " << force << endl;
	// 	cout << "force_fur1: " << force_fur1 << endl;
	// 	cout << "force_fur2: " << force_fur2 << endl;
	// }
	
	
	// comment this part to use the old approach//////////////////////////////////////////////////////////////

	return force;
	//return 1*force + 0.3*force_fur1 + 0.1*force_fur2;
	// return VectorXd::Zero(3);
}

//VectorXd getCurrentAnglesWithImpedanceAvoidance(VectorXd& currentX, VectorXd& currentV, VectorXd& desiredX, VectorXd& desiredV, VectorXd& currentCartCorrectiveVel, VectorXd& currentAngles, VectorXd& force, double timeStep) {
// VectorXd getCurrentAnglesWithImpedanceAvoidance(VectorXd currentAngles, VectorXd& currentV, VectorXd& desiredX, VectorXd& desiredV, double timeStep, VectorXd& inout_currentCartCorrectiveVel) {
// 	VectorXd currentX = kinematics.getEndEffectorPosition(currentAngles);


// 	VectorXd force = dynamicPotentialForce(currentX, desiredV); ///tau;		//force for obstacle avoidance

// 	VectorXd e = currentX - desiredX;		//error position p - pdesired
// 	VectorXd e_dot = currentV - desiredV;	//error in velocity v - vdesired

// 	MatrixXd M = MatrixXd::Identity(3,3);
// 	MatrixXd K = 150*MatrixXd::Identity(3,3);						//100
// 	MatrixXd D = 1.5*sqrt(4*K(1,1))*MatrixXd::Identity(3,3);      // 1.5 times of critical damping for smooth motion

// 	VectorXd e_dotdot = M.inverse()*(force - D*e_dot - K*e);	//gets calculated to minimize e

// 	inout_currentCartCorrectiveVel = integrateVector(inout_currentCartCorrectiveVel, e_dotdot, timeStep);

// 	MatrixXd J = kinematics.getJacobian(currentAngles);
// 	VectorXd currentCorrectiveVelocity = J.transpose()*inout_currentCartCorrectiveVel; 	//not pseudoInverse(J)??
// 	VectorXd currentAngleVelocities = kinematics.getCurrentVelocities(desiredX, desiredV, currentAngles, timeStep) + currentCorrectiveVelocity;
// 	currentAngleVelocities = currentAngleVelocities.transpose();	

// 	currentAngles = integrateVector(currentAngles, currentAngleVelocities, timeStep);

// 	return currentAngles;
// }

//VectorXd getCurrentAnglesWithImpedanceAvoidance(VectorXd& currentX, VectorXd& currentV, VectorXd& desiredX, VectorXd& desiredV, VectorXd& currentCartCorrectiveVel, VectorXd& currentAngles, VectorXd& force, double timeStep) {
void getCurrentAnglesWithImpedanceAvoidance(VectorXd& inout_currentAngles, VectorXd& currentV, VectorXd& desiredX, VectorXd& desiredV, double timeStep, bool correctVel, VectorXd& currentCartCorrectiveVel, 
											VectorXd& out_currentAngleVelocities) {
	
	VectorXd currentX = kinematics.getEndEffectorPosition(inout_currentAngles);


	VectorXd force = dynamicPotentialForce(currentX, desiredV); ///tau;		//force for obstacle avoidance

	VectorXd e = currentX - desiredX;		//error position p - pdesired
	VectorXd e_dot = currentV - desiredV;	//error in velocity v - vdesired

	MatrixXd M = MatrixXd::Identity(3,3);
	MatrixXd K = 150*MatrixXd::Identity(3,3);						//100
	MatrixXd D = 1.5*sqrt(4*K(1,1))*MatrixXd::Identity(3,3);      // 1.5 times of critical damping for smooth motion

	VectorXd e_dotdot = M.inverse()*(force - D*e_dot - K*e);	//gets calculated to minimize e

	currentCartCorrectiveVel = integrateVector(currentCartCorrectiveVel, e_dotdot, timeStep);

	MatrixXd J = kinematics.getJacobian(inout_currentAngles);
	VectorXd currentCorrectiveVel = J.transpose()*currentCartCorrectiveVel; 	//not pseudoInverse(J)??

	if (correctVel) {
		out_currentAngleVelocities = kinematics.getCurrentVelocities(desiredX, desiredV, inout_currentAngles, timeStep) + currentCorrectiveVel;
	}
	else {
		out_currentAngleVelocities = kinematics.getCurrentVelocities(desiredX, desiredV, inout_currentAngles, timeStep);
	}
	out_currentAngleVelocities = out_currentAngleVelocities.transpose();	
	inout_currentAngles = integrateVector(inout_currentAngles, out_currentAngleVelocities, timeStep);

	// //get new velocity with new current angles
	// J = kinematics.getJacobian(inout_currentAngles);
	// currentCorrectiveVel = J.transpose()*currentCartCorrectiveVel;
	// if (correctVel) {
	// 	out_currentAngleVelocities = kinematics.getCurrentVelocities(desiredX, desiredV, inout_currentAngles, timeStep) + currentCorrectiveVel;
	// }
	// else {
	// 	out_currentAngleVelocities = kinematics.getCurrentVelocities(desiredX, desiredV, inout_currentAngles, timeStep);
	// }
	// out_currentAngleVelocities = out_currentAngleVelocities.transpose();
	// inout_currentAngles = integrateVector(inout_currentAngles, out_currentAngleVelocities, timeStep);
}

//callback functions for all the subscribers, save data in global variables declared above
/*void getObstaclePosition(geometry_msgs::Point point) {
	obstaclePosition[0] = point.x;
	obstaclePosition[1] = point.y;
	obstaclePosition[2] = point.z;

	cout << "obstacle position callback " << endl << obstaclePosition << endl; 
}*/

/*void getLinkVelocity(geometry_msgs::Point msg) {
	x0d[0] = msg.x;
	x0d[1] = msg.y;
	x0d[2] = msg.z;
	//set x0d in kinematics class
	kinematics.setx0d(x0d);
}*/

/*void getClosestJoint(std_msgs::Float32 msg) {
	closestJoint = (int)msg.data;
	//set closest joint in kinematics class
	kinematics.setClosestJoint(closestJoint);
}*/

/*void getDistanceToJoint(geometry_msgs::Point point) {
	a = point.x;
	b = point.y;
	c = point.z;
}*/

/*void getObstacleVelocity(geometry_msgs::Point msg) {
	obstacleVelocity[0] = msg.x;
	obstacleVelocity[1] = msg.y;
	obstacleVelocity[2] = msg.z;
}*/

// void setCollisionStatus(std_msgs::Bool status) {
// 	collision = status.data;
// 	//cout << "collision: " << collision << endl;
// }

/*double handSubTime = 0;		//time between the position messages to compute the velocity
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
}*/

double handSubTime = 0;		//time between the position messages to compute the velocity
double preTime = 0;
//this callback function also writes to obstacle position for the force computation
void getTrackerPosition(qualisys::Subject msg) {				
	handSubTime = ros::Time::now().toSec() - preTime;		//the time since the last message
	preTime = ros::Time::now().toSec();

	obstacleVelocity[0] = (msg.position.x - obstaclePosition[0])/handSubTime;	//the difference between the last and the current position
	obstacleVelocity[1] = (msg.position.y - obstaclePosition[1])/handSubTime;
	obstacleVelocity[2] = (msg.position.z - obstaclePosition[2])/handSubTime;

	obstaclePosition[0] = msg.position.x;
	obstaclePosition[1] = msg.position.y;
	obstaclePosition[2] = msg.position.z;

	//std::cout << "tracker position: " << endl << obstaclePosition << std::endl;
	//std::cout << "tracker velocity: " << endl << obstacleVelocity << std::endl;

}

//robot eef pos in qualisys
void getRobotPosition(qualisys::Subject msg) {				

	robotPosition[0] = msg.position.x;
	robotPosition[1] = msg.position.y;
	robotPosition[2] = msg.position.z;
}

/*void callRobotFrame(qualisys::Subject msg) {
	robotBasePosition[0] = msg.position.x;
	robotBasePosition[1] = msg.position.y;
	robotBasePosition[2] = msg.position.z;
}*/

void getHumanTime(std_msgs::Float32 msg) {
	std::cout << "Human time received. Start proccessing..." << std::endl;
	if(msg.data < 4) {				//to avoid disturbing the results if one human movement was not detected, we assume the human takes max 4s to respond
		humanDuration = msg.data;
	}
	double end = ros::Time::now().toSec();
	if(end - startOfRobotMovement < 5 && end - startOfRobotMovement > 0.01) {
		humanTime = end - startOfRobotMovement;
	}
}

int nGoals = 3; //3;	//change this to the desired number of goals published from unity and everything will update automatically:)
VectorXd goalselect = VectorXd(nGoals); //


void getHumanGoal(std_msgs::Int32 msg) {
	int tempdata = msg.data;
	humanGoal = tempdata;
/*	int data = 0;
	switch (tempdata)
	{
		case 0: data = 3;
				break;
		case 1: data = 5;
				break;
		case 2: data = 7;
				break; 
	}

	humanGoal = 10; 	//default value that is always false, in case human goal does not match one of the existing
	for(int i = 0; i < nGoals; i++) {
		if(data == goalselect[i]) {
			humanGoal = i;
		}*/
	//}
	//cout << "goal out of totalGoals " << data << ", goal out of nGoals (3): " << humanGoal << endl;
	cout << "received goal value: " << humanGoal << endl;
}

void getPromp(const sensor_msgs::JointState::ConstPtr& msg)
{
	cout << "Prediction received. Start proccessing " << endl; 
	MatrixXd ptompData = MatrixXd::Zero(50,3);
    // string goal = msg->name[0];

    // if(goal == "0") humanGoal = 0;
    // if(goal == "1") humanGoal = 1;
    // if(goal == "2") humanGoal = 2;

    // std::cout << "human goal from promp " << humanGoal << ", original message " << goal << std::endl;

    for (int i = 0; i < 50; i++) {
        ptompData(i, 0) = msg->position[i];
        ptompData(i, 1) = msg->velocity[i];
        ptompData(i, 2) = msg->effort[i];
    }

    //cout << "Get all data" << endl; 
    // cout << "ptomp received " << endl;
    // geometry_msgs::Point futurePos; 
    futurePos1(0) = ptompData(25,0);	//one point in the future (chosen arbitrarily)
    futurePos1(1) = ptompData(25,1);
    futurePos1(2) = ptompData(25,2);

   //cout << "futurePos1: " << futurePos1 << endl; 

    futurePos2(0) = ptompData(48,0);	//one point in the future (chosen arbitrarily)
    futurePos2(1) = ptompData(48,1);
    futurePos2(2) = ptompData(48,2);

    //cout << "futurePos2: " << futurePos2 << endl; 
    // predictionPub.publish(futurePos);
    // ros::spinOnce();
}


void getcurRobotPos(const sensor_msgs::JointState::ConstPtr& msg)
{
	//extract information from the /joint_states topic
	// std::cout << "Received msgs from the robot..." << std::endl;
    for (int i = 0; i < 7; i++) {
        curKukabotPos(i) = msg->position[2*i];
        curKukabotVel(i) = msg->velocity[2*i];
        curKukabotAcc(i) = msg->effort[2*i];
    }


}


/*void getMotionStatus(std_msgs::Int32 msg) {
	//humanGoal = msg.data;
}*/

//due to lack of straight forward method to create 3d arrays, this function puts different matrices beside each other in a big matrix
void addToBigMatrix(MatrixXd& newMatrix, MatrixXd& bigMatrix, int ithMatrix) {
	//cout << "big " << bigMatrix.size() << endl;
	for(int i = 0; i < newMatrix.rows(); i++) {
		for(int j = 0; j < newMatrix.cols(); j++) {
			bigMatrix(i, j+ithMatrix*newMatrix.cols()) = newMatrix(i,j);
			//cout << "index " << j+ithMatrix*newMatrix.cols() << endl;
		}
	}
}

//global variable, stores all the angle velocities for each trajectory, angles stores all angles for each trajectory
MatrixXd angleVelocities, angles, angleAccelerations;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//contains the update loop, is the MOST IMPORTANT function, the core of this program
int executeTrajectories(DmpSolver* dmpSolver[], MatrixXd& goal, int nGoals, VectorXd& startJoint, ros::Rate& loop_rate, int nUpdates, int nSamplesPerUpdate, int subjectNr, int experimentNr);
//produces the same linear movement constantly
//int executeLinearTrajectories(DmpSolver* dmpSolver[], MatrixXd& goal, int nGoals, VectorXd& startJoint, ros::Rate& loop_rate, int nUpdates, int nSamplesPerUpdate, int subjectNr, int experimentNr);

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

#define MAXBUFSIZE  ((int) 1e6)

MatrixXd readMatrix(const char *filename)
    {
    int cols = 0, rows = 0;
    double buff[MAXBUFSIZE];

    // Read numbers from file into buffer.
    ifstream infile;
    infile.open(filename);
    cout << "Opening file: " << filename << endl;
    if (infile.is_open())
    	cout << "File opened." << endl;


    string line;
    double temp_data;
    //int ans = getline(infile, line);
    while (! infile.eof())	//! infile.eof()
        {
        //string line;
        getline(infile, line);

        int temp_cols = 0;

        stringstream stream(line);
        while(! stream.eof())
            stream >> buff[cols*rows+temp_cols++];

        if (temp_cols == 0)
            continue;

        if (cols == 0)
            cols = temp_cols;

        rows++;
        //ans = getline(infile, line);
        }

    infile.close();

    rows--;
    cout << "Finish reading. Prepare to save. Row = " << rows << "Col = " << cols << endl; 
    cout << buff << endl;
    // Populate matrix with numbers.
    MatrixXd result(rows,cols);

    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            result(i,j) = buff[ cols*i+j ];

    return result;
    };

////////////////////////////////////////////////////////////////////////////////////main///////////////////////////////////////////////////////////////////////////////////
ofstream distanceFile;
ros::Publisher endeffPub;
ros::Publisher forcePub;
ros::Publisher objvisibilityPub;

ros::Publisher pub_positions;
ros::Publisher pub_gains;
ros::Publisher pub_torque;

ros::Publisher pub_J[6];
ros::Publisher pub_intpositions;
ros::Publisher pub_posestampedpositions;
ros::Publisher pub_posestampedvelocities;
ros::Publisher pub_joint_traj_2Robot;



int main(int argc, char *argv[])
{
	std::cout << "Welcome to the experiment!" << std::endl;

	//define node
	ros::init(argc, argv, "dmp_ik");
	ros::NodeHandle node;

	//define publisher, Robot Publisher
	ros::Rate loop_rate(100);
	//these publishers work with the joint_impedance_controller
	pub_positions = node.advertise<std_msgs::Float64MultiArray>("/standing2/joint_impedance_controller/position", 100);
	pub_gains = node.advertise<std_msgs::Float64MultiArray>("/standing2/joint_impedance_controller/gains", 100);
	pub_torque = node.advertise<std_msgs::Float64MultiArray>("/standing2/joint_impedance_controller/torque", 100);
	objvisibilityPub = node.advertise<sensor_msgs::JointState>("/fromROS/objects/visibility",100);

	//array of publishers to publish each joint angle on a specific topic for unity/VR
	for(int i = 0; i < 6; i++) {
		int n = i+1;
		pub_J[i] = node.advertise<std_msgs::Int64>(str(boost::format("/standing2/joint_angles/J%1%") % n), 1);
	}
	pub_intpositions = node.advertise<std_msgs::String>("/standing2/joint_angles/position", 100);
	pub_posestampedpositions = node.advertise<geometry_msgs::PoseStamped>("/standing2/joint_angles/position_posestamped", 100);
	pub_posestampedvelocities = node.advertise<geometry_msgs::PoseStamped>("/standing2/joint_angles/velocity_posestamped", 100);

	initRosMsgs();

	//publisher to publish cartesian position of end effector (for debugging) 
	endeffPub = node.advertise<geometry_msgs::Point>("/positionEndeffector", 1);
	forcePub  = node.advertise<geometry_msgs::Point>("/force", 1);
	//publisher to publish the qualisys msgs to vrep
	//vrepPub = node.advertise<geometry_msgs::Point>("/wristForVrep", 100);
	//predictionPub = node.advertise<geometry_msgs::Point>("predictionForVrep", 100);
	//define subscriber to get obstacle position and closestPointToObstacle position (we need closest joint and distance to closest joint) from vrep
	//ros::Subscriber obsSub   = node.subscribe("/position/obstacle", 1, getObstaclePosition);	// --> this subscriber is for working with vrep
	//ros::Subscriber jointPosSub = node.subscribe("/x0d", 1, getLinkVelocity);
	//ros::Subscriber jointSub = node.subscribe("/closestJoint", 1, getClosestJoint);
	//ros::Subscriber distSub  = node.subscribe("/distanceToJoint", 1, getDistanceToJoint);
	//ros::Subscriber obsVelSub = node.subscribe("/obstacleVelocity", 1, getObstacleVelocity);
	//define subscriber for collision status from vrep
	//ros::Subscriber collSub = node.subscribe("/collision_status", 1, setCollisionStatus);		//not used
	//subscriber for qualisys (gets obstacle/hand position), gets positions from qualisys and sends them to vrep, vrep does computation and send info to code again
	//ros::Subscriber handSub = node.subscribe("/qualisys/wrist_right", 100, getHandPosition);		// subscriber to the camera system
	ros::Subscriber trackerSub = node.subscribe("/qualisys/humanpos", 100, getTrackerPosition);		//position of VRtracker
	ros::Subscriber robotPosSub = node.subscribe("/qualisys/roboteefpos", 100, getRobotPosition);
	handSubTime = ros::Time::now().toSec();
	//subscriber to get the time the human needs to reach the goal (from the prediction code)
	ros::Subscriber humanTimeSub = node.subscribe<std_msgs::Float32>("/motion_duration", 10, getHumanTime);
	ros::Subscriber getPrompSub = node.subscribe<sensor_msgs::JointState>("/JS_predict_wrist", 10, getPromp);
	ros::Subscriber humanGoalSub = node.subscribe<std_msgs::Int32>("/target_idx", 10, getHumanGoal);
	//ros::Subscriber humanGoalSub = node.subscribe<std_msgs::Int32>("/motion_status", 10, getMotionStatus);
	//subscriber to get the position of the base of the robot from the camera system
	//ros::Subscriber robotFrameSub = node.subscribe("/qualisys/standing2", 100, callRobotFrame);		//not used

	/////////////////////////////////////////////////////////////////////////
	//THIS DEFINES PUBLISHERS AND SUBSCRIBERS FOR CONTROLLING THE REAL ROBOT - KUKABOT
	ros::Subscriber getcurRobotPosSub = node.subscribe<sensor_msgs::JointState>("/joint_states", 10, getcurRobotPos);

	pub_joint_traj_2Robot = node.advertise<trajectory_msgs::JointTrajectory>("/joint_trajectory_controller/command",100);

////////////////////////////////////////////////////////////define start and goal position(s)///////////////////////////////////////////////////////////////
	//define goal and start
//	VectorXd startJoint(7); startJoint << -190.7 * PI / 180, -26.54 * PI / 180, -156.9* PI / 180, -81.03 * PI / 180, -115.88 * PI / 180, 0, 0;	//initial state in joint space (for training the policy)  //pre startjoint
	//VectorXd startJoint(7); startJoint << -190.7 * PI / 180, -10.54 * PI / 180, -166.9* PI / 180, -120.03 * PI / 180, -115.88 * PI / 180, 0, 0;	//initial state in joint space (for training the policy)
	//VectorXd startJoint(7); startJoint << 5.7 * PI / 180, -10.54 * PI / 180, -166.9* PI / 180, -120.03 * PI / 180, -115.88 * PI / 180, 0, 0;	//initial state in joint space (for training the policy)
	VectorXd startJoint(7); startJoint << -158.19 * PI / 180, -40.36 * PI / 180, -169.29* PI / 180, -100.24 * PI / 180, -12.04 * PI / 180, 0, 0;		//(for transferring the tasks)  newest one
	// VectorXd startJoint(7); startJoint << -158.19 * PI / 180, -40.36 * PI / 180, -156.29* PI / 180, -100.24 * PI / 180, -12.04 * PI / 180, 0, 0;		//(for transferring the tasks) testing real robot

	//Real configuration on the KUKA
    //startJoint << 1.07, 0.67, -0.3, -1.55, 0.18, 0.0, 0.0;   //J4 usually reaches the limit
    //startJoint << -0.71, 0.79, -2.00, 1.48, -0.75, -1.33, 0.0;
    //startJoint << 0.0, 0.67, -2.14, 1.59, 0.0, 0.0, 0.0;	// work pretty good
    startJoint << 0.197, 0.47, -1.94, 1.55, 0.0, 0.0, 0.0;


	VectorXd start(3); start = kinematics.getEndEffectorPosition(startJoint);													//initial state in cartesian space
	//std::cout << "start " << start << std::endl;

	// std::cout << "Would you like to change the starting configuration of the robot? (1 to change, 0 to keep)" << std::endl;
	// Bool ansChange;
	// cin >> ansChange;

	// if (ansChange)
	// {
	// 	std::cout << "Current configuration is [" << std::endl;

	// } 

	// //publish starting position to ROS once
	// geometry_msgs::PoseStamped msgposestart;
	// msgposestart.pose.position.x = startJoint[0]*180/PI; //first joint 
	// msgposestart.pose.position.y = startJoint[1]*180/PI; //2nd joint 
	// msgposestart.pose.position.z = startJoint[2]*180/PI; //3rd joint 
	// msgposestart.pose.orientation.x = startJoint[3]*180/PI; //4th joint
	// msgposestart.pose.orientation.y = startJoint[4]*180/PI; //5th joint
	// msgposestart.pose.orientation.z = startJoint[5]*180/PI; //6th joint
	// msgposestart.pose.orientation.w = 0; //
	// pub_posestampedpositions.publish(msgposestart);


	//////////////////////////////////////////////////////////////////////
	 MatrixXd goal(nGoals,3);
		 										//the number of goals, if changed add the new goal positions to the goal matrix
 	 // goal << 1.081, 0.447, 0.186,	//goal 1	//qualisys		//COMMENT OUT this part to use the goal positions from ros
			//   1.354, 0.377, 0.186,	//goal 2
			//   1.579, 0.548, 0.186;

 	 // goal <<  1.00, 0.00, 0.25,	//goal 1	//qualisys		//quick test on real robot
			//   1.00, 0.20, 0.25,	//goal 2
			//   1.00, 0.40, 0.25;

 	 goal <<  1.00, -0.10, 0.25,	//goal 1	//qualisys		//quick test on real robot
			  1.00, 0.10, 0.25,	//goal 2
			  1.00, 0.30, 0.25;

	// goal <<  1.34, 0.442, 0.270,	//goal 1	//qualisys		//quick test on real robot
	// 		  1.560, 0.442, 0.270,	//goal 2
	// 		  1.774, 0.442, 0.270;

	int totalGoals = 9; //total number of goals from Unity
	//wait for message for goal positions
	goalselect[0] = 4; goalselect[1] = 5; goalselect[2] = 6;   //here pick 3 goals that you want to run the experiment, global, needed for getHumanGoal
	//goalselect[0] = 2; goalselect[1] = 5; goalselect[2] = 8;
	//goalselect[0] = 3; goalselect[1] = 5; goalselect[2] = 7;
	// goalselect[0] = 2; goalselect[1] = 4; goalselect[2] = 9;

	// geometry_msgs::PoseStamped goal_msg[totalGoals];
	// cout << "Waiting for goal positions from Unity ..." << endl;
	// for(int i = 0; i < totalGoals; i++) {
	// 	goal_msg[i] = *(ros::topic::waitForMessage<geometry_msgs::PoseStamped>(str(boost::format("/fromUnity/goal%1%/pos") % (i+1))));
	// }

	// //MatrixXd goal(nGoals,3);

	// for(int i = 0; i < nGoals; i++) {
	// 	int temp = (int)goalselect[i];
	// 	goal(i,0) = -goal_msg[temp-1].pose.position.y;
	// 	goal(i,1) = goal_msg[temp-1].pose.position.x-0.05;
	// 	goal(i,2) = goal_msg[temp-1].pose.position.z;
	// }
	
	// cout << "received goals " << endl << goal << endl;
	
	///////////////////////////////////////////////////define the DMP///////////////////////////////////////////////////////////////////////////////////////////////
	double tau = 3;				//time constant
	double dt = 0.01;			//integration step duration
	double covarSize = 80; //250; 	//the bigger the variance, the bigger the sample space
	int n_basis_functions = 5;	//n gaussians per dmp

	int subjectNr = 10;			//specify here which policy should be loaded
	int experimentNr = 10;
	int updateNr = 13;

	cout << "How to initialize the dmp? (0 to initialize from scratch, 1 to train from trajectory, 2 to train from weights)" << endl;
	int answerLoadTraj;
	cin >> answerLoadTraj;

	DmpSolver *dmpSolver[nGoals]; 

	for(int i = 0; i < nGoals; i++) {
		VectorXd tempGoal = goal.block(i,0,1,3).transpose();
		if(answerLoadTraj == 0) {		//initialize new
			VectorXd dummyVec = VectorXd::Zero(4);
			Trajectory dummyTraj = Trajectory(dummyVec, dummyVec, dummyVec, dummyVec);
			VectorXd dummyWeights = VectorXd::Zero(3*n_basis_functions);
			dmpSolver[i] = new DmpSolver(tau, start, tempGoal, dt, dummyTraj, dummyWeights, covarSize, n_basis_functions);	//initialize dmps for all goals
		}
		else if(answerLoadTraj == 1){	//load saved trajectory
			cout << "Generating new DMPs with trained trajectory" << endl;
			//ifstream trajFile(str(boost::format("/home/hrcstudents/Documents/dmpbbo_ws/Results_v2/filelog/fulltraj_S%1%_E%2%_U%3%_G%4%_Sa%5%.txt") % subjectNr % experimentNr % updateNr % i % 0)); //load always sample zero
			MatrixXd input = readMatrix(str(boost::format("/home/hrcstudents/Documents/dmpbbo_ws/Results_v2/filelog/fulltraj_S%1%_E%2%_U%3%_G%4%_Sa%5%.txt") % subjectNr % experimentNr % updateNr % (i+1) % 1).c_str());
			cout << "File reading successful." << endl << input << endl;
			VectorXd ts  = input.block(0, 0, input.rows(), 1);
			MatrixXd y   = input.block(0, 1, input.rows(), 3);
			MatrixXd yd  = input.block(0, 4, input.rows(), 3);
			MatrixXd ydd = input.block(0, 7, input.rows(), 3);

			Trajectory traj = Trajectory(ts, y, yd, ydd);
			//Trajectory traj = Trajectory::generateMinJerkTrajectory(ts, start, tempGoal);
			cout << "Trajectory object generated. " << endl;
			VectorXd dummyWeights = VectorXd::Zero(3*n_basis_functions);
			dmpSolver[i] = new DmpSolver(tau, start, tempGoal, dt, traj, dummyWeights, covarSize, n_basis_functions, true);	//initialize dmps for all goals, "false" -> trajectory not used for training
		}
		else {			//load from weights
			VectorXd dummyVec = VectorXd::Zero(4);
			Trajectory dummyTraj = Trajectory(dummyVec, dummyVec, dummyVec, dummyVec);
			//MatrixXd weightsMatrix = readMatrix(str(boost::format("/home/hrcstudents/Documents/dmpbbo_ws/Results_v2/filelog_weights/weights_S%1%_E%2%_U%3%_G%4%.txt") % subjectNr % experimentNr % updateNr % (i+1)).c_str());
			//VectorXd weightsInput(Map<VectorXd>(weightsMatrix.data(), weightsMatrix.cols()*weightsMatrix.rows()));
			VectorXd weightsInput(15);
			switch(i) {
				// case 0: weightsInput << -9.1506,    6.9995,   -5.8735,   20.2522,    5.0957,   11.4351,   18.8189,   -9.5059,    0.9540,    5.1774,  -11.4106,  -20.6228,  -14.2249,  -11.0507,  -13.0173;	break;
				// case 1: weightsInput << -8.6976,   20.7057,   -5.6465,   33.7814,    4.8702,    7.7810,   14.6775,    1.3284,  -15.6979,   16.7648,  -12.8031,  -18.6501,  -17.7077,  -13.6133,  -16.1566;	break;
				// case 2: weightsInput << 2.9447,    3.3374,    4.1312,  -25.5429,   -6.1481,   10.9572,   -5.2967,  -10.1131,   31.8389,  -23.4505,  -13.7398,  -16.7666,   -6.9294,   -9.2702,   -8.6742;	break;
				//new case - 2,4,9 - robot opposite site
				// case 0: weightsInput << -2.7485,    6.8749,   -6.1089,    0.3560,    7.4983,   -4.8780,   -1.9247,    9.4527,   -1.3762,   0.1984,   -9.8409,   -12.3517,  -12.6774,  -10.2046,  -15.9868;	break;
				// case 1: weightsInput << -11.0915,   -8.3251,   -5.6050,  -15.7456,    3.2294,   -7.6069,   0.6662,    0.2431,    3.6066,  -11.9399,   -8.6627,  -10.4423,  -10.3893,  -11.4006,  -16.3322;	break;
				// case 2: weightsInput << 11.4734,   -6.8066,  -10.8245,    8.2235,    2.5498,    1.9741,   -1.2606,   -7.4825,  -14.5678,   22.3068,  -12.3253,  -17.4489,  -15.6204,  -12.8449,  -13.6262;	break;

				//new case - 2,4,9 - robot same site
				case 0: weightsInput << -4.0298,   17.7005,    0.7415,   12.3401,   1.9642,    2.9551,    9.7264,    5.6482,    3.0580,   -4.0411,  -13.0888,  -14.8400,  -14.3078,  -12.1872,   -13.7862;	break;
				case 1: weightsInput << -12.3728,    2.5005,    1.2454,   -3.7615,   -2.3047,    0.2261,   12.3172,   -3.5615,    8.0408,  -16.1794,  -11.9105,  -12.9306,  -12.0197,  -13.3832, -14.1317;	break;
				case 2: weightsInput << 10.1921,    4.0190,   -3.9741,   20.2076,   -2.9843,    9.8071,   10.3905,  -11.2870,  -10.1336,   18.0673,  -15.5731,  -19.9372,  -17.2508,  -14.8275,  -11.4256;	break;
				//test
				// case 0: weightsInput << -11.9350,   -4.7515,   -8.2871,  -10.7466,    3.8513,   -7.4124,    0.6514,    2.8704,    4.7022,   -8.4342,   -8.8622,  -11.7236,  -11.0481,  -10.6648,  -16.1390;	break;
				// case 1: weightsInput << 5.2747,  -12.7927,    0.8910,  -19.4528,    1.8329,   -3.8775,   -3.2495,   -3.2839,   -3.9606,  -10.0446,  -10.3552,   -9.8862,  -10.1998,  -12.8698,  -15.2347;	break;
				// case 2: weightsInput << 21.0123,   -4.8854,   -1.3624,   -7.7222,    2.0652,    0.6326,   -8.7381,    4.3734,   -5.7288,    1.6838,  -13.1355,  -13.8117,  -11.7676,  -11.3685,  -13.0809;	break;
			}
			dmpSolver[i] = new DmpSolver(tau, start, tempGoal, dt, dummyTraj, weightsInput, covarSize, n_basis_functions, false);	//initialize dmps for all goals, "false" -> trajectory not used for training
		}
	}

	cout << "model parameters: " << endl << dmpSolver[0]->getCentersWidthsAndWeights().transpose() << endl;
	////////////////////////////////////////////this part is needed if you change the covariance for task transfer/////////////
	int maxCost = 60;		//read this value from plot
	
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//cout << "Load the policy from file or train from scratch? (1 if load policy, 0 if train from scratch)" << endl;
	bool ansLoad, ansCovar;
	//cin >> ansLoad;
	ansLoad = false;
	bool continue_ = false;		//flag to stop continuing if statement if inner loop throws an exception

	if(ansLoad) {	      	  	//retrieve policy from file
		for(int i = 0; i<nGoals; i++) {
			string fileName = str(boost::format("/home/hrcstudents/Documents/dmpbbo_ws/Results_v2/filelog/trainedDmpPolicy_G%1%_S%2%_E%3%.txt") % i % subjectNr % experimentNr);	//goal 1
			ifstream ifs(fileName);
			// try {
				boost::archive::text_iarchive ia(ifs);
				ia >> dmpSolver[i];			//load stored dmpSolver data into current variable
			// }
			// catch(boost::archive::archive_exception) {
			// 	std::cout << "exception thrown. No policy found for the defined subject and experiment nr. Program will train a new policy." << std::endl;
			// 	cout << "loading for goal unsuccessful " << i << endl;
			// 	continue_ = true;	//even if the policy for only one of the goals was not found, all of them will be retrained
			// 	continue;
			// }
			// cout << "loading for goal successful " << i << endl;
			// dmpSolver[i]->_taskSolver->set_perturbation(0);	
			// dmpSolver[i]->_dmp->set_initial_state(start);	//here you can change the task, by either setting a new starting position with set_initial_state or by changing the goal positions with set_attractor_state
		}
	}
/*	if(ansLoad && !continue_) {		//if no exception was thrown, continue with the rest
		MatrixXd temp = dmpSolver[0]->_distribution->covar();	//diagonal matrix with covariance size as diagonal elements
		cout  << "current covariance (of the loaded policy)" << temp(0,0) << endl;
		double meanCost = dmpSolver[0]->_meanCost;		//mean cost of the previously trained policy
		cout << "mean cost of goal 1" << meanCost << endl;
		cout << "Change the covariance to " << covarSize << " or keep the current one? (1 if change, 0 if keep)" << endl;
		cin >> ansCovar;
		for(int i = 0; i < nGoals; i++) {
			if(ansCovar) {
				//double newCovarSize = meanCost/maxCost*covarSize;	//if the cost is still high keep the sampling variance high, else lower it proportionally		 				
				VectorXd meanInit = dmpSolver[0]->_meanInit;		
				//MatrixXd covarInit = newCovarSize * MatrixXd::Identity(meanInit.size(), meanInit.size());	//get initial covariance
				MatrixXd covarInit = covarSize * MatrixXd::Identity(meanInit.size(), meanInit.size());
				dmpSolver[i]->_distribution->set_covar(covarInit);								//reset noise
				cout << "Covariance changed to " << covarSize << endl;
			}
		}
	}	//else just train a policy from scratch*/

	cout << "Which subject is doing the experiment?" << endl;	//this information will be used to save the cost and angle files 
	cin >> subjectNr;
	cout << "Experiment number? (choose another value than the one used for training (" << experimentNr <<") to avoid overwriting)" << endl;
	cin >> experimentNr;
	//////////////////////////////////////////////////////habituation phase///////////////////////////////////////////////////////////////////////////////////////////
	//rollout one trajectory per goal for the habituation phase
	//problem here: the program as it is now will take the habituation results into account for the training, -> reset the dmps?
/*	int nUpdates = 1;		
    int nSamplesPerUpdate = 1;*/
    //uncomment the following line if you want to perform one trajectory to each goal first before starting the experiment
    //executeTrajectories(dmpSolver, goal, nGoals, startJoint, loop_rate, nUpdates, nSamplesPerUpdate, subjectNr, experimentNr);	
	// 	for(int i = 0; i < nGoals; i++) {
	// 	VectorXd tempGoal = goal.block(i,0,1,3).transpose();
	// 	dmpSolver[i] = new DmpSolver(tau, start, tempGoal, dt, covarSize);	//initialize dmps for all goals
	// }
    //cout << "habituation complete." << endl;
     /////////////////////////////////////////////////////run the policy improvement////////////////////////////////////////////////////////////////////////////////
    int nUpdates = 10; //15;
    int nSamplesPerUpdate = 5; //3;

    cout << "Press enter to start the experiment." << endl;
    cin.ignore();
    cin.ignore();
    ros::spinOnce();

    VectorXd starteefpos(3);
    VectorXd cureefpos(3);

    // bool switch_traj = 1;
    
 //    while (1)
	// {

	// if (switch_traj)
	// {
	// 	startJoint << 0.0, 0.67, -2.14, 1.59, 0.0, 0.0, 0.0;	
	// 	switch_traj = 0;
	// }	
	// else
	// {
	// 	startJoint << -0.71, 0.79, -2.00, 1.48, -0.75, -1.33, 0.0;
	// 	switch_traj = 1;
	// }

    //Here move the KUKA to the starting position
	bool START_POS_FLAG = 1;
	cureefpos = kinematics.getEndEffectorPosition(curKukabotPos);
	starteefpos = kinematics.getEndEffectorPosition(startJoint);


	std::cout << "Current position: " <<  cureefpos << std::endl;
	std::cout << "Home position: " <<  starteefpos << std::endl;
	
	VectorXd startKukaPos = VectorXd::Zero(7);		//error
	startKukaPos = curKukabotPos;

	trajectory_msgs::JointTrajectory KUKA_joint_state;   //msg to publish traj
	KUKA_joint_state.joint_names.resize(7);
	KUKA_joint_state.points.resize(1);

	std::vector<trajectory_msgs::JointTrajectoryPoint> points_n(1);

	points_n[0].positions.resize(7);
	points_n[0].velocities.resize(7);
	points_n[0].effort.resize(7);

	// points_n[1].positions.resize(7);
	// points_n[1].velocities.resize(7);
	// points_n[1].effort.resize(7);

	double start_time;					//start time
	int k_step;
	double traj_tsample = 0.01;

	
	start_time = ros::Time::now().toSec();
	k_step = 0;

	double d_time = 3; double time_msg_send = start_time; double duration_per_msg = 0;

	cout << "The robot is moving to the starting position..." << endl;
	while (START_POS_FLAG)
	{

		double tt =  ros::Time::now().toSec() - start_time;
		// cout << "Current time: " <<  tt << endl;
		while (tt > (k_step*traj_tsample))
		{
			k_step = k_step+1;
		}

		// Calculate next point
		double tt_next = (k_step+1)*traj_tsample;
		// double tt_nextnext = (k_step+3)*traj_tsample;

		if (tt_next<d_time)			//minimum jerk trajectory for smooth motion to the starting position 
		{
			for(int i = 0; i < 7; i++)
			{
				points_n[0].positions[i] = startKukaPos[i] + (startJoint[i] - startKukaPos[i])*(10*pow((tt_next/d_time),3) - 15*pow((tt_next/d_time),4) + 6*pow((tt_next/d_time),5));
				points_n[0].velocities[i] = (startJoint[i] - startKukaPos[i])*(30/d_time*pow((tt_next/d_time),2) - 60/d_time*pow((tt_next/d_time),3) + 30/d_time*pow((tt_next/d_time),4));

				// points_n[1].positions[i] = startKukaPos[i] + (startJoint[i] - startKukaPos[i])*(10*pow((tt_nextnext/d_time),3) - 15*pow((tt_nextnext/d_time),4) + 6*pow((tt_nextnext/d_time),5));
				// points_n[1].velocities[i] = (startJoint[i] - startKukaPos[i])*(30/d_time*pow((tt_nextnext/d_time),2) - 60/d_time*pow((tt_nextnext/d_time),3) + 30/d_time*pow((tt_nextnext/d_time),4));
				// points_n[0].velocities[i] = 0;
			}	
		
		}
		else
		{
			for(int i = 0; i < 7; i++)
			{
				points_n[0].positions[i] = startJoint[i];
				points_n[0].velocities[i] = 0;

				// points_n[1].positions[i] = startJoint[i];
				// points_n[1].velocities[i] = 0;
			}
			//cout << "Last data sent..." << endl;

		} 


		KUKA_joint_state.joint_names[0] = "kukabot_a1_joint";
		KUKA_joint_state.joint_names[1] = "kukabot_a2_joint"; 
		KUKA_joint_state.joint_names[2] = "kukabot_a3_joint";
		KUKA_joint_state.joint_names[3] = "kukabot_a4_joint";
		KUKA_joint_state.joint_names[4] = "kukabot_a5_joint";
		KUKA_joint_state.joint_names[5] = "kukabot_a6_joint";
		KUKA_joint_state.joint_names[6] = "kukabot_e1_joint";

		KUKA_joint_state.points[0] = points_n[0];
		// KUKA_joint_state.points[1] = points_n[1];

		KUKA_joint_state.points[0].positions[0] = points_n[0].positions[0];
		KUKA_joint_state.points[0].positions[1] = points_n[0].positions[1];
		KUKA_joint_state.points[0].positions[2] = points_n[0].positions[3];
		KUKA_joint_state.points[0].positions[3] = points_n[0].positions[4];
		KUKA_joint_state.points[0].positions[4] = points_n[0].positions[5];
		KUKA_joint_state.points[0].positions[5] = points_n[0].positions[6];
		KUKA_joint_state.points[0].positions[6] = points_n[0].positions[2];

		KUKA_joint_state.points[0].velocities[0] = points_n[0].velocities[0];
		KUKA_joint_state.points[0].velocities[1] = points_n[0].velocities[1];
		KUKA_joint_state.points[0].velocities[2] = points_n[0].velocities[3];
		KUKA_joint_state.points[0].velocities[3] = points_n[0].velocities[4];
		KUKA_joint_state.points[0].velocities[4] = points_n[0].velocities[5];
		KUKA_joint_state.points[0].velocities[5] = points_n[0].velocities[6];
		KUKA_joint_state.points[0].velocities[6] = points_n[0].velocities[2];

		// KUKA_joint_state.points[1].positions[0] = points_n[1].positions[0];
		// KUKA_joint_state.points[1].positions[1] = points_n[1].positions[1];
		// KUKA_joint_state.points[1].positions[2] = points_n[1].positions[3];
		// KUKA_joint_state.points[1].positions[3] = points_n[1].positions[4];
		// KUKA_joint_state.points[1].positions[4] = points_n[1].positions[5];
		// KUKA_joint_state.points[1].positions[5] = points_n[1].positions[6];
		// KUKA_joint_state.points[1].positions[6] = points_n[1].positions[2];

		// KUKA_joint_state.points[1].velocities[0] = points_n[1].velocities[0];
		// KUKA_joint_state.points[1].velocities[1] = points_n[1].velocities[1];
		// KUKA_joint_state.points[1].velocities[2] = points_n[1].velocities[3];
		// KUKA_joint_state.points[1].velocities[3] = points_n[1].velocities[4];
		// KUKA_joint_state.points[1].velocities[4] = points_n[1].velocities[5];
		// KUKA_joint_state.points[1].velocities[5] = points_n[1].velocities[6];
		// KUKA_joint_state.points[1].velocities[6] = points_n[1].velocities[2];

		// double temp_dur = tt_next - tt;
		KUKA_joint_state.points[0].time_from_start = ros::Duration(2*traj_tsample);
		// KUKA_joint_state.points[1].time_from_start = ros::Duration(4*traj_tsample);

		//check to see if the duration per message under threshold
		duration_per_msg = ros::Time::now().toSec() - time_msg_send;
		// cout << "Duration per message: " << duration_per_msg << endl;
		if (duration_per_msg > 0.015)
		{
			cout << "WARNING: duration per message is slow: " << duration_per_msg << endl;
		}
		time_msg_send = ros::Time::now().toSec();

		//publish message
		pub_joint_traj_2Robot.publish(KUKA_joint_state);
		ros::spinOnce();				//need if we have a subscribtion in the program
		loop_rate.sleep();

		//set the flag if the robot reaches the home position
		double errorPos=0;
		for(int i = 0; i < 7; i++)
		{
			errorPos = errorPos + abs(startJoint[i] - curKukabotPos[i]);
		}
		// std::cout << "Error Position: " << errorPos << std::endl;
		if (errorPos < 0.001)
		{
			START_POS_FLAG = 0;  //set the flag to get out of the while loop
		}
	} 

	cout << "The robot reached home position. Press enter to execute the trajectories..." << endl;
	cin.ignore();
    // cin.ignore();


	// }

	executeTrajectories(dmpSolver, goal, nGoals, startJoint, loop_rate, nUpdates, nSamplesPerUpdate, subjectNr, experimentNr);	//run the dmp, this function is the core of this program
	// executeLinearTrajectories(dmpSolver, goal, nGoals, startJoint, loop_rate, nUpdates, nSamplesPerUpdate, subjectNr, experimentNr);
	///////////////////////////////////////////////////////serialize (save) the current state of dmpSolver////////////////////////////////////////////////////////////
	//the policy is automatically saved, can be loaded when the program begins
	//name the serialization files differently if first time training or retraining
/*	for(int i = 0; i<nGoals; i++) {
		string fileName;
		if(ansLoad) {	
			fileName = str(boost::format("/home/hrcstudents/Documents/dmpbbo_ws/Results_v2/filelog/retrainedDmpPolicy_G%1%_S%2%_E%3%.txt") % i % subjectNr % experimentNr);	
		}
		else {
			fileName = str(boost::format("/home/hrcstudents/Documents/dmpbbo_ws/Results_v2/filelog/trainedDmpPolicy_G%1%_S%2%_E%3%.txt") % i % subjectNr % experimentNr);
		}
		ofstream ofs(fileName);
		boost::archive::text_oarchive oa(ofs);
		oa << dmpSolver[i];

		cout << "archive for goal successful " << i << endl;
	}	*/
	return 0;
}




////////////////////////////////////////////////////////////////////////////////publish messages//////////////////////////////////////////////////////////////////////////////////////////////////
void publishMessagesImpedance(VectorXd& angles, VectorXd& angleVelo, ros::Publisher& pub_positions, ros::Publisher& pub_torque, ros::Publisher& pub_gains){	//for joint_impedance_controller
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

    std_msgs::Int64MultiArray arraymsg;
    std_msgs::String strmsg;
    arraymsg.data.resize(6);

	for(int i = 0; i<6; i++) {
		std_msgs::Int64 msgf;
		msgf.data = (int)angles[i]*180/PI;
		pub_J[i].publish(msgf);

		arraymsg.data[i] = static_cast<int>(angles[i]*180/PI);
		
	}
	std::stringstream ss;
	ss <<"["<<angles[0]*180/PI<<","<<angles[1]*180/PI<<","<<angles[2]*180/PI<<","<<angles[3]*180/PI<<","<<angles[4]*180/PI<<","<<angles[5]*180/PI<<"]";
	strmsg.data = ss.str();
	pub_intpositions.publish(strmsg);

	// // testing with geometry_msgs
	// geometry_msgs::PoseStamped msgpose;
	// msgpose.pose.position.x = angles[0]*180/PI; //first joint 
	// msgpose.pose.position.y = angles[1]*180/PI; //2nd joint 
	// msgpose.pose.position.z = angles[2]*180/PI; //3rd joint 
	// msgpose.pose.orientation.x = angles[3]*180/PI; //4th joint
	// msgpose.pose.orientation.y = angles[4]*180/PI; //5th joint
	// msgpose.pose.orientation.z = angles[5]*180/PI; //6th joint
	// msgpose.pose.orientation.w = 0; //
	// pub_posestampedpositions.publish(msgpose);


	//publishing command to the kukabot
	trajectory_msgs::JointTrajectory KUKA_joint_state;   //msg to publish traj
	KUKA_joint_state.joint_names.resize(7);
	KUKA_joint_state.points.resize(1);

	std::vector<trajectory_msgs::JointTrajectoryPoint> points_n(1);

	points_n[0].positions.resize(7);
	points_n[0].velocities.resize(7);
	points_n[0].effort.resize(7);

	for(int i = 0; i < 7; i=i+2)
	{
		if (angles[i]>(167*PI/180))
		{
			points_n[0].positions[i] = 167*PI/180;
			std::cout << "Joint " << i+1 << " get close to limit: " << angles[i]*180/PI << ". Cut off at 167 degree." << std::endl;
		}
		else if (angles[i]<-(167*PI/180))
		{
				points_n[0].positions[i] = -167*PI/180;
				std::cout << "Joint " << i+1 << " get close to limit: " << angles[i]*180/PI << ". Cut off at -167 degree." << std::endl;
		}
			else
				points_n[0].positions[i] = angles[i];
	}

	for(int i = 1; i < 7; i=i+2)
	{
		if (angles[i]>(117*PI/180))
		{
			points_n[0].positions[i] = 117*PI/180;
			std::cout << "Joint " << i+1 << " get close to limit: " << angles[i]*180/PI << ". Cut off at 117 degree." << std::endl;
		}
		else if (angles[i]<-(117*PI/180))
		{
				points_n[0].positions[i] = -117*PI/180;
				std::cout << "Joint " << i+1 << " get close to limit: " << angles[i]*180/PI << ". Cut off at -117 degree." << std::endl;
		}
			else
				points_n[0].positions[i] = angles[i];
	}

	for(int i = 0; i < 7; i=i+1)
	{
		points_n[0].velocities[i] = angleVelo[i];
	}

	geometry_msgs::PoseStamped msgpose; geometry_msgs::PoseStamped msgvelo; 
	msgpose.header.stamp = ros::Time::now();
	msgpose.pose.position.x = points_n[0].positions[0]; //first joint 
	msgpose.pose.position.y = points_n[0].positions[1]; //2nd joint 
	msgpose.pose.position.z = points_n[0].positions[2]; //3rd joint 
	msgpose.pose.orientation.x = points_n[0].positions[3]; //4th joint
	msgpose.pose.orientation.y = points_n[0].positions[4]; //5th joint
	msgpose.pose.orientation.z = points_n[0].positions[5]; //6th joint
	msgpose.pose.orientation.w = points_n[0].positions[6]; //
	pub_posestampedpositions.publish(msgpose);

	msgvelo.header.stamp = ros::Time::now();
	msgvelo.pose.position.x = points_n[0].velocities[0]; //first joint 
	msgvelo.pose.position.y = points_n[0].velocities[1]; //2nd joint 
	msgvelo.pose.position.z = points_n[0].velocities[2]; //3rd joint 
	msgvelo.pose.orientation.x = points_n[0].velocities[3]; //4th joint
	msgvelo.pose.orientation.y = points_n[0].velocities[4]; //5th joint
	msgvelo.pose.orientation.z = points_n[0].velocities[5]; //6th joint
	msgvelo.pose.orientation.w = points_n[0].velocities[6]; //
	pub_posestampedvelocities.publish(msgvelo);

	// KUKA_joint_state.header.stamp = ros::Time::now();

	KUKA_joint_state.joint_names[0] = "kukabot_a1_joint";
	KUKA_joint_state.joint_names[1] = "kukabot_a2_joint"; 
	KUKA_joint_state.joint_names[2] = "kukabot_a3_joint";
	KUKA_joint_state.joint_names[3] = "kukabot_a4_joint";
	KUKA_joint_state.joint_names[4] = "kukabot_a5_joint";
	KUKA_joint_state.joint_names[5] = "kukabot_a6_joint";
	KUKA_joint_state.joint_names[6] = "kukabot_e1_joint";

	KUKA_joint_state.points[0] = points_n[0];

	KUKA_joint_state.points[0].positions[0] = points_n[0].positions[0];
	KUKA_joint_state.points[0].positions[1] = points_n[0].positions[1];
	KUKA_joint_state.points[0].positions[2] = points_n[0].positions[3];
	KUKA_joint_state.points[0].positions[3] = points_n[0].positions[4];
	KUKA_joint_state.points[0].positions[4] = points_n[0].positions[5];
	KUKA_joint_state.points[0].positions[5] = points_n[0].positions[6];
	KUKA_joint_state.points[0].positions[6] = points_n[0].positions[2];

	KUKA_joint_state.points[0].velocities[0] = points_n[0].velocities[0];
	KUKA_joint_state.points[0].velocities[1] = points_n[0].velocities[1];
	KUKA_joint_state.points[0].velocities[2] = points_n[0].velocities[3];
	KUKA_joint_state.points[0].velocities[3] = points_n[0].velocities[4];
	KUKA_joint_state.points[0].velocities[4] = points_n[0].velocities[5];
	KUKA_joint_state.points[0].velocities[5] = points_n[0].velocities[6];
	KUKA_joint_state.points[0].velocities[6] = points_n[0].velocities[2];

	KUKA_joint_state.points[0].time_from_start = ros::Duration(2*0.01);

	pub_joint_traj_2Robot.publish(KUKA_joint_state);
}
///////////////////////////////////////////////////////////////////////////cost function/////////////////////////////////////////////////////////////////////////////////////////////////////////
void computeCosts(MatrixXd& ys, MatrixXd& yds, MatrixXd& ydds, int nDims, VectorXd& cost, VectorXd& ts, bool collisionOccured, MatrixXd& angles, MatrixXd& angleVelocities, MatrixXd& ydReal, double robot_time, bool accuracy, MatrixXd& y1, MatrixXd& y2) {
	//maybe add another cost that the dmp has to stay within the limits of the robot (within a cartesian radius)
	// double wHumanT = 25, wRobot = 0, wAccuracy = 15, wJoint = 1, wEndeffector = 5, wTrajectory = 3, wHumanD = 0;		//weights to make all variables in the same range
	// double human_time = 1, /*accuracy = 0,*/ joint_jerk = 1, endeffector_jerk = 1, trajectory = 0, human_duration = 0;
	double wHumanT = 8, wRobot = 0, wAccuracy = 10, wJoint = 1, wEndeffector = 1, wTrajectory = 0, wHumanD = 1, wTrajectory_distance = 3;		//weights to make all variables in the same range
	double human_time = 1, /*accuracy = 0,*/ joint_jerk = 1, endeffector_jerk = 1, trajectory = 0, human_duration = 0, trajectory_distance = 0;
	double deltaT = ts[1] - ts[0];

	for(int i = 0; i < cost.size(); i++) {
		cost[i] = 0;
	}

	// cout << "Was the prediction correct? (1 if correct, 0 if incorrect) " << endl;
	// //cin >> accuracy; 
	accuracy = !accuracy;	//if it was correct -> no cost, if it was incorrect -> cost = 1

	//difference between the direct output of the dmp and the deformed trajectory, over time the dmp should get closer to the deformed trajectory
	VectorXd posEndEffDmp(3), posEndeffReal(3);
	for(int t = 0; t < ts.size(); t++) {
		posEndEffDmp[0] = ys(t,0);
		posEndEffDmp[1] = ys(t,1);
		posEndEffDmp[2] = ys(t,2);

		VectorXd currentAngles = angles.block(t,0,1,7).transpose();
		posEndeffReal = kinematics.getEndEffectorPosition(currentAngles);
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

	//punish trajectories that are too close, trajectories that are too close with a small ts (beginning of movement) produce high cost
	VectorXd offset = VectorXd::Ones(ts.size())*0.01;		//to avoid dividing by 0
	trajectory_distance = (sqrt((ys - y1).rowwise().squaredNorm().array())/((ts).array()+offset.array())).sum()		// ||ys-y1||/ts
						+ (sqrt((ys - y2).rowwise().squaredNorm().array())/((ts).array()+offset.array())).sum();




	human_time = humanTime - human_duration;
	human_duration = humanDuration; //normalized duration of human movement

	cout << "COST VALUES: " << endl;
	cost[1] = human_time*wHumanT;	
	cout << "human time " << human_time << endl;	
	cost[2] = robot_time*wRobot;				//not effective, as it is always the same
	cost[3] = accuracy*wAccuracy;
	cout << "accuracy " << cost[3] << endl;
	//cost[4] = joint_jerk*wJoint;
	cost[4] = (joint_jerk-5500)/(700000-120000)*wJoint;
	cout << "joint cost " << cost[4] << endl;
	//cost[5] = endeffector_jerk*wEndeffector;
	cost[5] = (endeffector_jerk-2000)/(200000-40000)*wEndeffector;
	cout << "endeffector cost " << cost[5] << endl;
	//cost[6] = wCollision*collisionOccured;
	cost[6] = wTrajectory*(trajectory-1)/5;
	cout << "trajectory cost " << cost[6] << endl;
	cost[7] = wHumanD*human_duration;
	cout << "human duration " << human_duration << endl;	

	cost[8] = wTrajectory_distance*((1/trajectory_distance)-0.003)/0.006637;
	cout << "trajectory_distance " << cost[8] << endl;

	cost[0] = cost.sum();
	//cout << "Computing cost finished " << endl;

}
//////////////////////////////////////////////////////////////////////////////////////////update loop//////////////////////////////////////////////////////////////////////
int executeTrajectories(DmpSolver* dmpSolver[], MatrixXd& goal, int nGoals, VectorXd& startJoint, ros::Rate& loop_rate, int nUpdates, int nSamplesPerUpdate, int subjectNr, int experimentNr) {
	ofstream myfile; 		//for saving the cost results in a file for plotting 
	//define some values
    VectorXd ts = dmpSolver[0]->_ts;		//vector of time steps for the trajectory
    //cout << "ts " << ts << endl;
    int nDims = dmpSolver[0]->_nDims;
    int nBasisFunc = dmpSolver[0]->_nBasisFunc;
    double timeStep = ts[1] - ts[0];
    double tau = dmpSolver[0]->_tau;

    angleVelocities = MatrixXd::Zero(ts.size(), 7);				//initialization
	angles = MatrixXd::Zero(ts.size(), 7);
	angleAccelerations = MatrixXd::Zero(ts.size(),7);
    
    VectorXd costs(9); //has the detailed costs, gets filled by compute costs, costs[0] is the totat cost

    MatrixXd samples = MatrixXd::Zero(nSamplesPerUpdate, nDims*nBasisFunc), costVariables = MatrixXd::Zero(nSamplesPerUpdate, (4*nDims+1)*ts.size());
    MatrixXd allSamples = MatrixXd::Zero(samples.rows(), samples.cols()*nGoals), allCostVariables = MatrixXd::Zero(costVariables.rows(), costVariables.cols()*nGoals);

    int meanCost = 0;
    // int counter = 0;

    VectorXd currentAngles(7); currentAngles = startJoint;
	VectorXd currentAngleVelocities = VectorXd::Zero(7), currentAcceleration = VectorXd::Zero(7);
	MatrixXd correctiveVelocity = MatrixXd::Zero(ts.size(),7);	//get integrated from the potential force
	VectorXd currentCorrectiveVelocity = VectorXd::Zero(7);
	MatrixXd realYd = MatrixXd::Zero(ts.size(),3);		//the cartesian velocities of the deformed trajectory (for cost computation

	VectorXd currentCartCorrectiveVel = VectorXd::Zero(3);
	VectorXd currentCartCorrectivePos = VectorXd::Zero(3);
	MatrixXd cart_correctivePos = MatrixXd::Zero(ts.size(), 3);
	VectorXd posEndEff = VectorXd::Zero(3);


	for(int iUpdate = 0; iUpdate < nUpdates; iUpdate++) {
		//////////Here define the scene (objects) that will be visible in Unity////////////////////////////
		// if (iUpdate==0)
		// {
		sensor_msgs::JointState obj_vis;
		obj_vis.position.resize(9);
		obj_vis.position[0] = 0; obj_vis.position[1] = 0; obj_vis.position[2] = 0;
		obj_vis.position[3] = 1; obj_vis.position[4] = 1; obj_vis.position[5] = 1;
		obj_vis.position[6] = 0; obj_vis.position[7] = 0; obj_vis.position[8] = 0;

		objvisibilityPub.publish(obj_vis);
		ros::spinOnce();
		// }
		// if (iUpdate==7)
		// {
		// 	sensor_msgs::JointState obj_vis;
		// 	obj_vis.position.resize(9);
		// 	obj_vis.position[0] = 0; obj_vis.position[1] = 1; obj_vis.position[2] = 0;
		// 	obj_vis.position[3] = 1; obj_vis.position[4] = 1; obj_vis.position[5] = 1;
		// 	obj_vis.position[6] = 0; obj_vis.position[7] = 1; obj_vis.position[8] = 0;

		// 	objvisibilityPub.publish(obj_vis);
		// 	ros::spinOnce();
		// }
		// if (iUpdate==14)
		// {
		// 	sensor_msgs::JointState obj_vis;
		// 	obj_vis.position.resize(9);
		// 	obj_vis.position[0] = 0; obj_vis.position[1] = 1; obj_vis.position[2] = 0;
		// 	obj_vis.position[3] = 1; obj_vis.position[4] = 1; obj_vis.position[5] = 1;
		// 	obj_vis.position[6] = 1; obj_vis.position[7] = 0; obj_vis.position[8] = 1;

		// 	objvisibilityPub.publish(obj_vis);
		// 	ros::spinOnce();
		// }
		//////////////////////////////End of definition////////////////////////////////////////////////////
    	cout << "+++++++++++++++++++++++++++++++++++++++++ Update Nr: " << iUpdate << " +++++++++++++++++++++++++++++++++++++++++" << endl;

		MatrixXd y, yd, ydd, y1 = MatrixXd::Zero(ts.size(), nDims), y2 = MatrixXd::Zero(ts.size(), nDims);								//the trajectory values that will be published

    	int nCostVariables = 4*nDims+1; 			//y, yd, ydd, ts, forcing terms
    	MatrixXd row(1, costVariables.cols()), row1, row2;		//temporary variable
    	MatrixXd rollout, rollout1, rollout2;							//to store the rollout data for each sample
    	MatrixXd costPerSample(nSamplesPerUpdate, nGoals);
    	MatrixXd shortCostVariables = MatrixXd::Zero(costVariables.rows(), costVariables.cols()), shortCostVariables1, shortCostVariables2;

    	for(int i = 0; i < nGoals; i++) {
	    	// //the lines here are to save the updated trajectory before sampling
	    	MatrixXd row_meanTraj, costVariables_meanTraj;
	    	dmpSolver[i]->_taskSolver->performRollouts(dmpSolver[i]->_distribution->mean().transpose(), costVariables_meanTraj);
	    	row_meanTraj = costVariables_meanTraj.row(0);
	    	MatrixXd rollout_meanTraj = (Map<MatrixXd>(row_meanTraj.data(), nCostVariables, ts.size())).transpose();
	    	y = rollout_meanTraj.block(0, 0, ts.size(), nDims);				//unmodified dmp outcome
	    	yd = rollout_meanTraj.block(0, nDims, ts.size(), nDims);
	    	writeToFileDmpEndEff(subjectNr, experimentNr, iUpdate, i+1, 0, y);	//used to train a new dmp with it
	    	// //end
	    	dmpSolver[i]->generateSamples(nSamplesPerUpdate, samples);		//you generate multiple samples
	    	dmpSolver[i]->performRollouts(samples, costVariables);			//costVariables now includes y_out, yd_out, ydd_out, ts, forcing terms for each sample
    		addToBigMatrix(samples, allSamples, i);							//save values in a big matrix, needed for the update
	    	addToBigMatrix(costVariables, allCostVariables, i);
	    }
	    cout << "test" << endl;

    	//sample loop, 
    	VectorXd s = VectorXd::Zero(nGoals);	//vector of counters for all samples for all goals
    	while(s.sum() < nSamplesPerUpdate*nGoals) {	//while (all samples have not been executed) any element of s is below the sample amount	
			
			cout << "Press enter to roll out trajectory." << endl;
			cin.ignore();
			
			Vector3d currentGoal;
    		int a = rand() % nGoals;	//random number between 0 and nGoals-1
    		if(s[a] < nSamplesPerUpdate) {	//make sure we execute all different samples for all goals
    			cout << "+++++++++++++++++++++ targeted goal: " << a+1 << " +++++++++++++++++++++" << endl;
	    		shortCostVariables = allCostVariables.block(0, costVariables.cols()*a, costVariables.rows(), costVariables.cols());	//chose a random block of the costVariables to execute
				currentGoal = goal.block(a,0,1,3).transpose();

				//the following if else is defining variables that will be needed in the cost computation for the distances between trajectories.
				if (a == 0) {
					shortCostVariables1 = allCostVariables.block(0, costVariables.cols()*1, costVariables.rows(), costVariables.cols());
					shortCostVariables2 = allCostVariables.block(0, costVariables.cols()*2, costVariables.rows(), costVariables.cols());
				}
				else if (a == 1) {
					shortCostVariables1 = allCostVariables.block(0, costVariables.cols()*0, costVariables.rows(), costVariables.cols());
					shortCostVariables2 = allCostVariables.block(0, costVariables.cols()*2, costVariables.rows(), costVariables.cols());
				}
				else {
					shortCostVariables1 = allCostVariables.block(0, costVariables.cols()*0, costVariables.rows(), costVariables.cols());
					shortCostVariables2 = allCostVariables.block(0, costVariables.cols()*1, costVariables.rows(), costVariables.cols());
				}
				row1 = shortCostVariables1.row(s[a]);
				row2 = shortCostVariables2.row(s[a]);
				rollout1 = (Map<MatrixXd>(row1.data(), nCostVariables, ts.size())).transpose();
				rollout2 = (Map<MatrixXd>(row2.data(), nCostVariables, ts.size())).transpose();
				y1 = rollout1.block(0, 0, ts.size(), nDims);
				y2 = rollout2.block(0, 0, ts.size(), nDims);
    		}
    		else {
    			continue;		//go to the end of the loop and try another a 
    		}
    		//cout << "Press enter to roll out trajectory." << endl;
    		//cin.ignore();

    		bool collisionOccured = 0;

    		row = shortCostVariables.row(s[a]);
			rollout = (Map<MatrixXd>(row.data(), nCostVariables, ts.size())).transpose();
			//cout << "rollout " << endl << rollout << endl;
    		y = rollout.block(0, 0, ts.size(), nDims);				//unmodified dmp outcome 
    		writeToFileDmpEndEff(subjectNr, experimentNr, iUpdate, a+1, s[a]+1, y);
    		yd = rollout.block(0, nDims, ts.size(), nDims);
    		ydd = rollout.block(0, 2*nDims, ts.size(), nDims);
    		writeToFileTrajectory(subjectNr, experimentNr, iUpdate, a+1, s[a]+1, ts, y, yd, ydd);	//used to train a new dmp with it
    		//time loop/execution loop

    		bool usingKDL = 0;    // use kdl for inverse kinematic solver

    		double duration = 0; //duration for one trajectory

			angles.block(0,0,1,7) = currentAngles.transpose();	//save it in the big matrix

			//ADD CHANGE HERE TO SEND THE TRAJ TO THE REAL ROBOT------------------------------------
// 			double start_time = ros::Time::now().toSec();
// 			int k_step = 0;

// 			while ((ros::Time::now().toSec() - start_time) < tau)		//duration of the traj is tau
// 			{
// 				if(k_step == 0) {										//start to measure the robot movement time
//     				startOfRobotMovement = ros::Time::now().toSec();
//     			}
//     			double tt = ros::Time::now().toSec() - start_time;		//current time 
//     			while (ts(k_step) < tt)		// check the current time to select the datapoint from dmp
//     			{
//     				k_step = k_step + 1;
//     				if (k_step > (ts.size()-1))		//end of the dmp data
//     				{
//     					k_step = ts.size()-1;
//     				}
//     			}
//     			//next data point send to the robot, go for 1 step further
//     			double tt_next = ts(k_step+1);
//     			VectorXd currentY = kinematics.getEndEffectorPosition(currentAngles);
// 	    		VectorXd currentV = VectorXd::Zero(3);
// 	    		VectorXd desiredX = y.row(k_step+1).transpose();
// 	    		VectorXd desiredV = yd.row(k_step+1).transpose();

// 	    		currentV = realYd.block(k_step,0,1,3).transpose();		//HAS TO BE CHANGED (t-1)

// 	    		timeStep = tt_next - tt;
// 		    	getCurrentAnglesWithImpedanceAvoidance(currentAngles, currentV, desiredX, desiredV, timeStep, currentCartCorrectiveVel);	//return value not used, only currentCartCorrectiveVel gets written

// 		    	MatrixXd J = kinematics.getJacobian(currentAngles);
// 				currentCorrectiveVelocity = J.transpose()*currentCartCorrectiveVel;

// 				//currentCorrectiveVelocity is the end effector avoidance
// 				currentAngleVelocities = kinematics.getCurrentVelocities(desiredX, desiredV, currentAngles, timeStep) + currentCorrectiveVelocity;	//save the new velocities (without modification) in the variable angleVelocities

// 				angleVelocities.block(k_step+1,0,1,7) = currentAngleVelocities.transpose();
// 				realYd.block(k_step+1,0,1,3).transpose() = J*currentAngleVelocities;

// 				currentAngles = integrateVector(currentAngles, currentAngleVelocities, timeStep);
// //					posEndEff = kinematics.getEndEffectorPosition(currentAngles);
// 				if ((k_step+1) != (ts.size()-1)) {
// 					angles.block(k_step+1,0,1,7).transpose() = currentAngles;
// 					posEndEff = kinematics.getEndEffectorPosition(currentAngles);
// 				}
	
// 				posEndEff = y.row(t).transpose();

// 			}

			//ADD CHANGE HERE TO SEND THE TRAJ TO THE REAL ROBOT------------------------------------

    		for(int t = 0; t < ts.size()-2; t++) {

    			if(t == 0) {
    				startOfRobotMovement = ros::Time::now().toSec();
    			}
	    		//currentY = y.row(t);
	    		VectorXd currentY = kinematics.getEndEffectorPosition(currentAngles);
	    		VectorXd currentV = VectorXd::Zero(3);
	    		VectorXd desiredX = y.row(t).transpose();
	    		VectorXd desiredV = yd.row(t).transpose();

	    		if (t !=0) {
	    			VectorXd currentV = realYd.block(t-1,0,1,3).transpose();		//HAS TO BE CHANGED (t-1)
	    		}

	    		//VectorXd inout_currentAngles, VectorXd& currentV, VectorXd& desiredX, VectorXd& desiredV, double timeStep, bool correctVel, VectorXd& currentCartCorrectiveVel, VectorXd& out_currentAngleVelocities)
		    	getCurrentAnglesWithImpedanceAvoidance(currentAngles, currentV, desiredX, desiredV, timeStep, true, currentCartCorrectiveVel, currentAngleVelocities);	//return value not used, only currentCartCorrectiveVel gets written
		    	//currentAngles = getCurrentAnglesWithImpedanceAvoidance(currentAngles, currentV, desiredX, desiredV, timeStep, currentCartCorrectiveVel);
		    	//cout << "called impedance wd" << endl;
//		    	posEndEff = y.row(t).transpose();

		    	if(usingKDL) {
		    	}
		    	else {
					MatrixXd J = kinematics.getJacobian(currentAngles); 
					//currentCorrectiveVelocity = J.transpose()*currentCartCorrectiveVel;	//old

					//currentCorrectiveVelocity is the end effector avoidance, old
					//currentAngleVelocities = kinematics.getCurrentVelocities(desiredX, desiredV, currentAngles, timeStep) + currentCorrectiveVelocity;	//save the new velocities (without modification) in the variable angleVelocities

					angleVelocities.block(t,0,1,7) = currentAngleVelocities.transpose();	//new
					realYd.block(t,0,1,3).transpose() = J*currentAngleVelocities;

					//currentAngles = integrateVector(currentAngles, currentAngleVelocities, timeStep); //old
//					posEndEff = kinematics.getEndEffectorPosition(currentAngles);
					if (t != (ts.size()-1)) {
						angles.block(t,0,1,7).transpose() = currentAngles;
						posEndEff = kinematics.getEndEffectorPosition(currentAngles);
					}
		
					posEndEff = y.row(t).transpose();
				}

	    		//publish end effector cartesian position (for testing)
				geometry_msgs::Point pos;
				pos.x = posEndEff[0];
				pos.y = posEndEff[1];
				pos.z = posEndEff[2];
				endeffPub.publish(pos);
				// ros::spinOnce();

				//checkConstraints(currentAngles);	//check if the angles fit the constraints before publishing

				publishMessagesImpedance(currentAngles, currentAngleVelocities, pub_positions, pub_torque, pub_gains);
				ros::spinOnce();				//need if we have a subscribtion in the program
				loop_rate.sleep();

				if(t == ts.size()-3) {
					duration = ros::Time::now().toSec() - startOfRobotMovement;
					cout << "duration for this robot trajectory: " << duration << endl;
				}
			}
			bool accuracy = (humanGoal == a); 	//is goal should come from /motion_status, a is the goal number for this rollout (0,1,2), this should be given as a parameter to computeCosts
			humanGoal = 5; 		//reset the value before the next callback function
			computeCosts(y, yd, ydd, nDims, costs, ts, collisionOccured, angles, angleVelocities, realYd, duration, accuracy, y1, y2);	
    		costPerSample(s[a],a) = costs[0];

    		//int subjectNr, int experimentNr, int updateNr, int goalNr, int sampleNr, VectorXd& data
    		writeToFile(subjectNr, experimentNr, iUpdate, a+1, s[a]+1, costs);
    		writeToFileAngles(subjectNr, experimentNr, iUpdate, a+1, s[a]+1, angles);
    		//cout << "finish write to file" << endl;

//    		cin.ignore(); 	//pause before going back
    		double counter = ros::Time::now().toSec();
    		while(ros::Time::now().toSec()-counter < 0.2) {		//until 0.4s keep force activated and running to get to the final goal position

    			//Reset the prediction variables since the human should reach the goal already
	    		futurePos1 = Vector3d::Zero();
	    		futurePos2 = Vector3d::Zero();

    			int t = ts.size()-3;
	    		VectorXd currentY = kinematics.getEndEffectorPosition(currentAngles);	//current end effector position
	    		VectorXd currentV = realYd.block(t,0,1,3).transpose();					//current endeffector velocity
	    		VectorXd desiredX = y.row(t).transpose();
	    		VectorXd desiredV = yd.row(t).transpose();

	    		getCurrentAnglesWithImpedanceAvoidance(currentAngles, currentV, desiredX, desiredV, timeStep, true, currentCartCorrectiveVel, currentAngleVelocities);
				//getCurrentAnglesWithImpedanceAvoidance(currentAngles, currentV, desiredX, desiredV, timeStep, currentCartCorrectiveVel);
				//MatrixXd J = kinematics.getJacobian(currentAngles);
				//currentCorrectiveVelocity = J.transpose()*currentCartCorrectiveVel;
				//currentAngleVelocities = kinematics.getCurrentVelocities(desiredX, desiredV, currentAngles, timeStep); // + currentCorrectiveVelocity;
	
				publishMessagesImpedance(currentAngles, currentAngleVelocities, pub_positions, pub_torque, pub_gains);
				ros::spinOnce();				//need if we have a subscribtion in the program
				loop_rate.sleep();
    		}

    		//cout << "time out. prepare to go back to start position" << endl;

    		///cout << "Press enter to continue." << endl;
    		//cin.ignore();
    		//bring the robot back to the start position (publish message3 in reversed order)
			for(int tReverse = ts.size()-3; tReverse >= 0; tReverse--) {

				//Reset the prediction variables since the human should reach the goal already
	    		futurePos1 = Vector3d::Zero();
	    		futurePos2 = Vector3d::Zero();

				MatrixXd J = kinematics.getJacobian(currentAngles);
				VectorXd currentY = kinematics.getEndEffectorPosition(currentAngles);
				VectorXd currentV = J*currentAngleVelocities;
	    		VectorXd desiredX = y.row(tReverse).transpose();
	    		VectorXd desiredV = -yd.row(tReverse).transpose();

	    		getCurrentAnglesWithImpedanceAvoidance(currentAngles, currentV, desiredX, desiredV, timeStep, true, currentCartCorrectiveVel, currentAngleVelocities);
	    		//getCurrentAnglesWithImpedanceAvoidance(currentAngles, currentV, desiredX, desiredV, timeStep, currentCartCorrectiveVel);
	    		//J = kinematics.getJacobian(currentAngles);
				//currentCorrectiveVelocity = J.transpose()*currentCartCorrectiveVel;
				//currentAngleVelocities = kinematics.getCurrentVelocities(desiredX, desiredV, currentAngles, timeStep); // + currentCorrectiveVelocity;

				//VectorXd endEff = kinematics.getEndEffectorPosition(currentAngles);	//currentAngles 
				VectorXd endEff = desiredX;
				geometry_msgs::Point pos;
				pos.x = endEff[0];
				pos.y = endEff[1];
				pos.z = endEff[2];
				endeffPub.publish(pos);
//				ros::spinOnce();  only need to call once per loop
				publishMessagesImpedance(currentAngles, currentAngleVelocities, pub_positions, pub_torque, pub_gains);
				ros::spinOnce();				//need if we have a subscribtion in the program
				loop_rate.sleep();	
			}

			//cout << "finish going back. wait for 1s" << endl;
			counter = ros::Time::now().toSec();       //pause 1s before continue but still has potential field effect

			while(ros::Time::now().toSec()-counter < 0.2)
			{

				//Reset the prediction variables since the human should reach the goal already
	    		futurePos1 = Vector3d::Zero();
	    		futurePos2 = Vector3d::Zero();


	    		//currentAngles = startJoint;		
				MatrixXd J = kinematics.getJacobian(currentAngles);
				VectorXd currentY = kinematics.getEndEffectorPosition(currentAngles);
				VectorXd currentV = J*currentAngleVelocities;
				VectorXd desiredX =  y.row(0).transpose();// kinematics.getEndEffectorPosition(startJoint); //y.row(0).transpose();
				VectorXd desiredV = VectorXd::Zero(3);

				getCurrentAnglesWithImpedanceAvoidance(currentAngles, currentV, desiredX, desiredV, timeStep, true, currentCartCorrectiveVel, currentAngleVelocities);
				//getCurrentAnglesWithImpedanceAvoidance(currentAngles, currentV, desiredX, desiredV, timeStep, currentCartCorrectiveVel);
				//J = kinematics.getJacobian(currentAngles);
				//currentCorrectiveVelocity = J.transpose()*currentCartCorrectiveVel;
				//currentAngleVelocities = kinematics.getCurrentVelocities(desiredX, desiredV, currentAngles, timeStep);

				publishMessagesImpedance(currentAngles, currentAngleVelocities, pub_positions, pub_torque, pub_gains);
				ros::spinOnce();				//need if we have a subscribtion in the program
				loop_rate.sleep();

			}
			cout << "finish waiting. prepare to start a new one" << endl;
			// currentAngles = startJoint;		
			// currentAngles = startJoint;		
			// currentAngleVelocities = VectorXd::Zero(7);						
			// publishMessagesImpedance(currentAngles, currentAngleVelocities, pub_positions, pub_torque, pub_gains);		// correct the starting configuration of the robot beforing starting a new one
			// ros::spinOnce();				//need if we have a subscribtion in the program
			// loop_rate.sleep();

			s[a]++; 	//keep track of which samples we executed
			// counter++;
    	}

    	for(int i = 0; i < nGoals; i++) {
    		MatrixXd shortSamples = allSamples.block(0, samples.cols()*i, samples.rows(), samples.cols());	
    		VectorXd shortCostPerSample = (VectorXd)costPerSample.block(0,i, costPerSample.rows(), 1);
    		dmpSolver[i]->updateDistribution(shortSamples, shortCostPerSample);
		}

		for(int i = 0; i < nGoals; i++) {	//print and save the weight values for each update
			cout << "weights for goal " << i << endl;
			VectorXd weights = dmpSolver[i]->getCentersWidthsAndWeights();
			cout << weights << endl;
			writeToFileWeights(subjectNr, experimentNr, iUpdate, i+1, weights);
		}

    	for(int i = 0; i < costPerSample.rows(); i++) {		//print total cost for each sample after every update
			cout << "costPerSample: " << costPerSample(i,0) << ", " << flush;		//index has to be corrected
			meanCost += costPerSample(i,0);
    	}
    	meanCost /= costPerSample.rows();
    	cout << "\n" << endl;

		if(iUpdate == 3 || iUpdate == 6) {
			cout << "Time for a break!" << endl;
			cin.ignore();			//pause the experiment to give time for a questionnaire
		}
	}		//end update loop
	cout << "You are done!" << endl;
	return meanCost;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void writeToFile(int subjectNr, int experimentNr, int updateNr, int goalNr, int sampleNr, VectorXd& data) {
	string fileName = str(boost::format("/home/hrcstudents/Documents/dmpbbo_ws/Results_v2/filelog/S%1%_E%2%_U%3%_G%4%_Sa%5%.txt") % subjectNr % experimentNr % updateNr % goalNr % sampleNr);
	ofstream file;
	file.open(fileName);
	file << data;
	file.close();
}

void writeToFileAngles(int subjectNr, int experimentNr, int updateNr, int goalNr, int sampleNr, MatrixXd& data) {
	string fileName = str(boost::format("/home/hrcstudents/Documents/dmpbbo_ws/Results_v2/filelog/angles_S%1%_E%2%_U%3%_G%4%_Sa%5%.txt") % subjectNr % experimentNr % updateNr % goalNr % sampleNr);
	ofstream file;
	file.open(fileName);
	file << data;
	file.close();
}

void writeToFileDmpEndEff(int subjectNr, int experimentNr, int updateNr, int goalNr, int sampleNr, MatrixXd& data) {
	string fileName = str(boost::format("/home/hrcstudents/Documents/dmpbbo_ws/Results_v2/filelog/dmptraj_S%1%_E%2%_U%3%_G%4%_Sa%5%.txt") % subjectNr % experimentNr % updateNr % goalNr % sampleNr);
	ofstream file;
	file.open(fileName);
	file << data;
	file.close();
}

void writeToFileTrajectory(int subjectNr, int experimentNr, int updateNr, int goalNr, int sampleNr, VectorXd& ts, MatrixXd& y, MatrixXd& yd, MatrixXd& ydd) {
	MatrixXd data(ts.rows(), ts.cols() + y.cols() + yd.cols() + ydd.cols());
	data << ts, y, yd, ydd;
	string fileName = str(boost::format("/home/hrcstudents/Documents/dmpbbo_ws/Results_v2/filelog/fulltraj_S%1%_E%2%_U%3%_G%4%_Sa%5%.txt") % subjectNr % experimentNr % updateNr % goalNr % sampleNr);
	ofstream file;
	file.open(fileName);
	file << data;
	file.close();
}

void writeToFileWeights(int subjectNr, int experimentNr, int updateNr, int goalNr, VectorXd& data) {
	string fileName = str(boost::format("/home/hrcstudents/Documents/dmpbbo_ws/Results_v2/filelog_weights/weights_S%1%_E%2%_U%3%_G%4%.txt") % subjectNr % experimentNr % updateNr % goalNr);
	ofstream file;
	file.open(fileName);
	file << data;
	file.close();
}
// void checkConstraints(VectorXd& angles) {
// 	//check if these angles fit the kuka constraints and fix them
// 	double const170 = 165*PI/180;	//+/-170 degrees is the limit for joint 0,2,4,6
// 	double const120 = 115*PI/180;	//+/-120 degrees is the limit for joint 1,3,5
// 	for(int i = 0; i < 7; i +=2) {
// 		if(angles[i] < -const170) {
// 			std::cout << "out of bounds, angle: " << i+1 << ", value: " << angles[i]*180/PI << std::endl;
// 			angles[i] = -const170;
// 		}
// 		if(angles[i] > const170) {
// 			std::cout << "out of bounds, angle: " << i+1 << ", value: " << angles[i]*180/PI << std::endl;
// 			angles[i] = const170;
// 		}
// 	}
// 	for(int i = 1; i < 7; i +=2) {
// 		if(angles[i] < -const120) {
// 			std::cout << "out of bounds, angle: " << i+1 << ", value: " << angles[i]*180/PI << std::endl;
// 			angles[i] = -const120;
// 		}
// 		if(angles[i] > const120) {
// 			std::cout << "out of bounds, angle: " << i+1 << ", value: " << angles[i]*180/PI << std::endl;
// 			angles[i] = const120;
// 		}
// 	}
// }

//void integrateVector(VectorXd& prevValue, VectorXd& derivative, double timeStep, VectorXd& result){
VectorXd integrateVector(VectorXd& prevValue, VectorXd& derivative, double timeStep){
	return prevValue + derivative*timeStep;
}

 //  	MetaParametersRBFN* meta_parameters = new MetaParametersRBFN(1,n_basis_functions,intersection_height);
 //  	ModelParametersRBFN* model_parameters = new ModelParametersRBFN(centers,widths,weights);

 //  	// FunctionApproximatorRBFN* fa_rbfn = new FunctionApproximatorRBFN(meta_parameters);
	
	// vector<FunctionApproximator*> function_approximators1(n_dims);
	// vector<FunctionApproximator*> function_approximators2(n_dims);
	// vector<FunctionApproximator*> function_approximators3(n_dims);
 //  	for (int i_dim=0; i_dim<n_dims; i_dim++)
 //  		function_approximators1[i_dim] = new FunctionApproximatorRBFN(model_parameters);  
	// for (int i_dim=0; i_dim<n_dims; i_dim++)
 //  		function_approximators2[i_dim] = new FunctionApproximatorRBFN(model_parameters);
 //  	for (int i_dim=0; i_dim<n_dims; i_dim++)
 //  		function_approximators3[i_dim] = new FunctionApproximatorRBFN(model_parameters);

 //  	VectorXd one_1 = VectorXd::Ones(1);
 //    VectorXd one_0 = VectorXd::Zero(1);
 //    DynamicalSystem *goal_system1= new ExponentialSystem(tau,start,goal.row(1),15);
 //    DynamicalSystem *goal_system2= new ExponentialSystem(tau,start,goal.row(2),15);
 //    DynamicalSystem *goal_system3= new ExponentialSystem(tau,start,goal.row(3),15);

 //    DynamicalSystem *phase_system= new TimeSystem(tau,false);
 //    DynamicalSystem *gating_system=new ExponentialSystem(tau,one_1,one_0,5);


 //    Dmp* dmp1 = new Dmp(tau, start, goal.row(1),function_approximators1, alpha_spring_damper,  goal_system1, phase_system,  gating_system);
	// Dmp* dmp2 = new Dmp(tau, start, goal.row(2),function_approximators2, alpha_spring_damper,  goal_system2, phase_system,  gating_system);
	// Dmp* dmp3 = new Dmp(tau, start, goal.row(3),function_approximators3, alpha_spring_damper,  goal_system3, phase_system,  gating_system);

	// // Make the task solver
 //  	set<string> parameters_to_optimize;
 //  	parameters_to_optimize.insert("weights");

 //  	bool use_normalized_parameter=false;
 //  	TaskSolverDmp* task_solver1 = new TaskSolverDmp(dmp1,parameters_to_optimize,
 //                                       dt,integrate_dmp_beyond_tau_factor,use_normalized_parameter);
 //  	TaskSolverDmp* task_solver2 = new TaskSolverDmp(dmp2,parameters_to_optimize,
 //                                       dt,integrate_dmp_beyond_tau_factor,use_normalized_parameter);
 //  	TaskSolverDmp* task_solver3 = new TaskSolverDmp(dmp3,parameters_to_optimize,
 //                                       dt,integrate_dmp_beyond_tau_factor,use_normalized_parameter);

 //  	MatrixXd xs_ana1, xds_ana1, forcing_terms_ana1, fa_output_ana1;
 //  	MatrixXd xs_ana2, xds_ana2, forcing_terms_ana2, fa_output_ana2;
 //  	MatrixXd xs_ana3, xds_ana3, forcing_terms_ana3, fa_output_ana3;

 //  	dmp1->analyticalSolution(time_steps,xs_ana1,xds_ana1,forcing_terms_ana1,fa_output_ana1);
 //  	dmp2->analyticalSolution(time_steps,xs_ana2,xds_ana2,forcing_terms_ana2,fa_output_ana2);
 //  	dmp3->analyticalSolution(time_steps,xs_ana3,xds_ana3,forcing_terms_ana3,fa_output_ana3);

 //  	MatrixXd output_ana1(time_steps.size(),1+xs_ana1.cols()+xds_ana1.cols());
 //  	MatrixXd output_ana2(time_steps.size(),1+xs_ana2.cols()+xds_ana2.cols());
 //  	MatrixXd output_ana3(time_steps.size(),1+xs_ana3.cols()+xds_ana3.cols());

 //  	output_ana1 << xs_ana1, xds_ana1, time_steps;
 //  	output_ana2 << xs_ana2, xds_ana2, time_steps;
 //  	output_ana3 << xs_ana3, xds_ana3, time_steps;

 //  	// Make the initial distribution
 //  	VectorXd mean_init1;
 //  	VectorXd mean_init2;
 //  	VectorXd mean_init3;

 //  	dmp1->getParameterVectorSelected(mean_init1);
 //  	dmp2->getParameterVectorSelected(mean_init2);
 //  	dmp3->getParameterVectorSelected(mean_init3);

 //  	MatrixXd covar_init1 = covarSize*MatrixXd::Identity(mean_init1.size(),mean_init1.size());
 //  	MatrixXd covar_init2 = covarSize*MatrixXd::Identity(mean_init2.size(),mean_init2.size());
 //  	MatrixXd covar_init3 = covarSize*MatrixXd::Identity(mean_init3.size(),mean_init3.size());

 //  	DistributionGaussian* distribution1 = new DistributionGaussian(mean_init1,covar_init1);
 //  	DistributionGaussian* distribution2 = new DistributionGaussian(mean_init2,covar_init2);
 //  	DistributionGaussian* distribution3 = new DistributionGaussian(mean_init3,covar_init3);

 //  	// Make the parameter updater
	// double eliteness = 10;
	// double covar_decay_factor = 0.83;
	// string weighting_method("PI-BB");
	// Updater* updater1 = new UpdaterCovarDecay(eliteness, covar_decay_factor, weighting_method);
	// Updater* updater2 = new UpdaterCovarDecay(eliteness, covar_decay_factor, weighting_method);
	// Updater* updater3 = new UpdaterCovarDecay(eliteness, covar_decay_factor, weighting_method);

/*int executeLinearTrajectories(DmpSolver* dmpSolver[], MatrixXd& goal, int nGoals, VectorXd& startJoint, ros::Rate& loop_rate, int nUpdates, int nSamplesPerUpdate, int subjectNr, int experimentNr)
{
	ofstream myfile; 		//for saving the cost results in a file for plotting 
	//define some values
    VectorXd ts = dmpSolver[0]->_ts;		//vector of time steps for the trajectory
    //cout << "ts " << ts << endl;
    int nDims = dmpSolver[0]->_nDims;
    int nBasisFunc = dmpSolver[0]->_nBasisFunc;
    double timeStep = ts[1] - ts[0];
    double tau = dmpSolver[0]->_tau;

    angleVelocities = MatrixXd::Zero(ts.size(), 7);				//initialization
	angles = MatrixXd::Zero(ts.size(), 7);
	angleAccelerations = MatrixXd::Zero(ts.size(),7);
    
    VectorXd costs(9); //has the detailed costs, gets filled by compute costs, costs[0] is the totat cost

    int meanCost = 0;
    // int counter = 0;

    MatrixXd samples = MatrixXd::Zero(nSamplesPerUpdate, nDims*nBasisFunc), costVariables = MatrixXd::Zero(nSamplesPerUpdate, (4*nDims+1)*ts.size());
    VectorXd currentAngles(7); currentAngles = startJoint;
	VectorXd currentAngleVelocities = VectorXd::Zero(7), currentAcceleration = VectorXd::Zero(7);
	MatrixXd correctiveVelocity = MatrixXd::Zero(ts.size(),7);	//get integrated from the potential force
	VectorXd currentCorrectiveVelocity = VectorXd::Zero(7);
	MatrixXd realYd = MatrixXd::Zero(ts.size(),3);		//the cartesian velocities of the deformed trajectory (for cost computation

	VectorXd currentCartCorrectiveVel = VectorXd::Zero(3);
	VectorXd currentCartCorrectivePos = VectorXd::Zero(3);
	MatrixXd cart_correctivePos = MatrixXd::Zero(ts.size(), 3);
	VectorXd posEndEff = VectorXd::Zero(3);


	MatrixXd row_meanTraj, costVariables_meanTraj0, costVariables_meanTraj1, costVariables_meanTraj2;
	dmpSolver[0]->_taskSolver->performRollouts(dmpSolver[0]->_distribution->mean().transpose(), costVariables_meanTraj0);
	dmpSolver[1]->_taskSolver->performRollouts(dmpSolver[1]->_distribution->mean().transpose(), costVariables_meanTraj1);
	dmpSolver[2]->_taskSolver->performRollouts(dmpSolver[2]->_distribution->mean().transpose(), costVariables_meanTraj2);

	for(int iUpdate = 0; iUpdate < nUpdates; iUpdate++) {
		sensor_msgs::JointState obj_vis;
		obj_vis.position.resize(9);
		obj_vis.position[0] = 0; obj_vis.position[1] = 0; obj_vis.position[2] = 0;
		obj_vis.position[3] = 1; obj_vis.position[4] = 1; obj_vis.position[5] = 1;
		obj_vis.position[6] = 0; obj_vis.position[7] = 0; obj_vis.position[8] = 0;

		objvisibilityPub.publish(obj_vis);
		ros::spinOnce();
		//////////////////////////////End of definition////////////////////////////////////////////////////
    	cout << "+++++++++++++++++++++++++++++++++++++++++ Update Nr: " << iUpdate << " +++++++++++++++++++++++++++++++++++++++++" << endl;

		MatrixXd y, yd, ydd, y1 = MatrixXd::Zero(ts.size(), nDims), y2 = MatrixXd::Zero(ts.size(), nDims);								//the trajectory values that will be published

    	int nCostVariables = 4*nDims+1; 			//y, yd, ydd, ts, forcing terms
    	MatrixXd row(1, costVariables.cols()), row1, row2;		//temporary variable
    	MatrixXd rollout, rollout1, rollout2;							//to store the rollout data for each sample
    	MatrixXd costPerSample(nSamplesPerUpdate, nGoals);
    	MatrixXd shortCostVariables, shortCostVariables1, shortCostVariables2;


    	//sample loop, 
    	VectorXd s = VectorXd::Zero(nGoals);	//vector of counters for all samples for all goals
    	while(s.sum() < nSamplesPerUpdate*nGoals) {	//while (all samples have not been executed) any element of s is below the sample amount	
			
	    	// //the lines here are to save the updated trajectory before sampling
	    	
			
			Vector3d currentGoal;
    		int a = rand() % nGoals;	//random number between 0 and nGoals-1
    		if(s[a] < nSamplesPerUpdate) {	//make sure we execute all different samples for all goals
    			cout << "+++++++++++++++++++++ targeted goal: " << a+1 << " +++++++++++++++++++++" << endl;
		    	
		    	switch(a) {
		    		case 0: shortCostVariables = costVariables_meanTraj0; break;
		    		case 1: shortCostVariables = costVariables_meanTraj1; break;
		    		case 2: shortCostVariables = costVariables_meanTraj2; break;
		    	}

				currentGoal = goal.block(a,0,1,3).transpose();
    		}
    		else {
    			continue;		//go to the end of the loop and try another a 
    		}
    		//cout << "Press enter to roll out trajectory." << endl;
    		//cin.ignore();

    		bool collisionOccured = 0;

    		row = shortCostVariables.row(0);
			rollout = (Map<MatrixXd>(row.data(), nCostVariables, ts.size())).transpose();
			//cout << "rollout " << endl << rollout << endl;
    		y = rollout.block(0, 0, ts.size(), nDims);				//unmodified dmp outcome 
    		writeToFileDmpEndEff(subjectNr, experimentNr, iUpdate, a+1, s[a]+1, y);
    		yd = rollout.block(0, nDims, ts.size(), nDims);
    		ydd = rollout.block(0, 2*nDims, ts.size(), nDims);
    		writeToFileTrajectory(subjectNr, experimentNr, iUpdate, a+1, s[a]+1, ts, y, yd, ydd);	//used to train a new dmp with it
    		//time loop/execution loop


    		double duration = 0; //duration for one trajectory

			angles.block(0,0,1,7) = currentAngles.transpose();	//save it in the big matrix

    		for(int t = 0; t < ts.size()-2; t++) {

    			if(t == 0) {
    				startOfRobotMovement = ros::Time::now().toSec();
    			}
	    		//currentY = y.row(t);
	    		VectorXd currentY = kinematics.getEndEffectorPosition(currentAngles);
	    		VectorXd currentV = VectorXd::Zero(3);
	    		VectorXd desiredX = y.row(t).transpose();
	    		VectorXd desiredV = yd.row(t).transpose();

	    		if (t !=0) {
	    			VectorXd currentV = realYd.block(t-1,0,1,3).transpose();		//HAS TO BE CHANGED (t-1)
	    		}


		    	//getCurrentAnglesWithImpedanceAvoidance(currentAngles, currentV, desiredX, desiredV, timeStep, currentCartCorrectiveVel);	//return value not used, only currentCartCorrectiveVel gets written
	    		getCurrentAnglesWithImpedanceAvoidance(currentAngles, currentV, desiredX, desiredV, timeStep, true, currentCartCorrectiveVel, currentAngleVelocities);
	    		cout << "called impedance" << endl;

	//		    	posEndEff = y.row(t).transpose();

		    	if(usingKDL) {
		    		currentCartCorrectivePos = integrateVector(currentCartCorrectivePos, currentCartCorrectiveVel, timeStep);
			    	currentAngles = kdl.iksolverLevenberg(y.row(t).transpose() + 0.01*currentCartCorrectivePos, MatrixXd::Zero(3,3), currentAngles);
			    	posEndEff = kinematics.getEndEffectorPosition(currentAngles);
		    	}
		    	else {
					MatrixXd J = kinematics.getJacobian(currentAngles);
					//currentCorrectiveVelocity = J.transpose()*currentCartCorrectiveVel;

					//currentCorrectiveVelocity is the end effector avoidance
					//currentAngleVelocities = kinematics.getCurrentVelocities(desiredX, desiredV, currentAngles, timeStep) + currentCorrectiveVelocity;	//save the new velocities (without modification) in the variable angleVelocities

					angleVelocities.block(t,0,1,7) = currentAngleVelocities;
					realYd.block(t,0,1,3).transpose() = J*currentAngleVelocities.transpose();

					//currentAngles = integrateVector(currentAngles, currentAngleVelocities, timeStep);
	//					posEndEff = kinematics.getEndEffectorPosition(currentAngles);
					if (t != (ts.size()-1)) {
						angles.block(t,0,1,7).transpose() = currentAngles;
						posEndEff = kinematics.getEndEffectorPosition(currentAngles);
					}
		
					posEndEff = y.row(t).transpose();
				}

	    		//publish end effector cartesian position (for testing)
				geometry_msgs::Point pos;
				pos.x = posEndEff[0];
				pos.y = posEndEff[1];
				pos.z = posEndEff[2];
				endeffPub.publish(pos);
				ros::spinOnce();

				//checkConstraints(currentAngles);	//check if the angles fit the constraints before publishing

				publishMessagesImpedance(currentAngles, currentAngleVelocities, pub_positions, pub_torque, pub_gains);
				ros::spinOnce();				//need if we have a subscribtion in the program
				loop_rate.sleep();

				if(t == ts.size()-3) {
					duration = ros::Time::now().toSec() - startOfRobotMovement;
					cout << "duration for this robot trajectory: " << duration << endl;
				}
			}
			bool accuracy = (humanGoal == a); 	//is goal should come from /motion_status, a is the goal number for this rollout (0,1,2), this should be given as a parameter to computeCosts
			humanGoal = 5; 		//reset the value before the next callback function
			computeCosts(y, yd, ydd, nDims, costs, ts, collisionOccured, angles, angleVelocities, realYd, duration, accuracy, y1, y2);	
    		costPerSample(s[a],a) = costs[0];

    		//int subjectNr, int experimentNr, int updateNr, int goalNr, int sampleNr, VectorXd& data
    		writeToFile(subjectNr, experimentNr, iUpdate, a+1, s[a]+1, costs);
    		writeToFileAngles(subjectNr, experimentNr, iUpdate, a+1, s[a]+1, angles);
    		//cout << "finish write to file" << endl;

	//    		cin.ignore(); 	//pause before going back
    		double counter = ros::Time::now().toSec();
    		while(ros::Time::now().toSec()-counter < 0.2) {		//until 0.4s keep force activated and running to get to the final goal position

    			//Reset the prediction variables since the human should reach the goal already
	    		futurePos1 = Vector3d::Zero();
	    		futurePos2 = Vector3d::Zero();

    			int t = ts.size()-3;
	    		VectorXd currentY = kinematics.getEndEffectorPosition(currentAngles);	//current end effector position
	    		VectorXd currentV = realYd.block(t,0,1,3).transpose();					//current endeffector velocity
	    		VectorXd desiredX = y.row(t).transpose();
	    		VectorXd desiredV = yd.row(t).transpose();

				//currentAngles = getCurrentAnglesWithImpedanceAvoidance(currentAngles, currentV, desiredX, desiredV, timeStep, currentCartCorrectiveVel);
				getCurrentAnglesWithImpedanceAvoidance(currentAngles, currentV, desiredX, desiredV, timeStep, false, currentCartCorrectiveVel, currentAngleVelocities);
	
				publishMessagesImpedance(currentAngles, currentAngleVelocities, pub_positions, pub_torque, pub_gains);
				ros::spinOnce();				//need if we have a subscribtion in the program
				loop_rate.sleep();
    		}

    		//cout << "time out. prepare to go back to start position" << endl;

    		///cout << "Press enter to continue." << endl;
    		//cin.ignore();
    		//bring the robot back to the start position (publish message3 in reversed order)
			for(int tReverse = ts.size()-3; tReverse >= 0; tReverse--) {

				//Reset the prediction variables since the human should reach the goal already
	    		futurePos1 = Vector3d::Zero();
	    		futurePos2 = Vector3d::Zero();

				MatrixXd J = kinematics.getJacobian(currentAngles);
				VectorXd currentY = kinematics.getEndEffectorPosition(currentAngles);
				VectorXd currentV = J*currentAngleVelocities;
	    		VectorXd desiredX = y.row(tReverse).transpose();
	    		VectorXd desiredV = -yd.row(tReverse).transpose();

	    		//currentAngles = getCurrentAnglesWithImpedanceAvoidance(currentAngles, currentV, desiredX, desiredV, timeStep, currentCartCorrectiveVel);
	    		getCurrentAnglesWithImpedanceAvoidance(currentAngles, currentV, desiredX, desiredV, timeStep, true, currentCartCorrectiveVel, currentAngleVelocities);

				//VectorXd endEff = kinematics.getEndEffectorPosition(currentAngles);	//currentAngles 
				VectorXd endEff = desiredX;
				geometry_msgs::Point pos;
				pos.x = endEff[0];
				pos.y = endEff[1];
				pos.z = endEff[2];
				endeffPub.publish(pos);
	//				ros::spinOnce();  only need to call once per loop
				publishMessagesImpedance(currentAngles, currentAngleVelocities, pub_positions, pub_torque, pub_gains);
				ros::spinOnce();				//need if we have a subscribtion in the program
				loop_rate.sleep();	
			}

			//cout << "finish going back. wait for 1s" << endl;
			counter = ros::Time::now().toSec();       //pause 1s before continue but still has potential field effect
			while(ros::Time::now().toSec()-counter < 0.2)
			{
				//Reset the prediction variables since the human should reach the goal already
	    		futurePos1 = Vector3d::Zero();
	    		futurePos2 = Vector3d::Zero();

				MatrixXd J = kinematics.getJacobian(currentAngles);
				VectorXd currentY = kinematics.getEndEffectorPosition(currentAngles);
				VectorXd currentV = J*currentAngleVelocities;
				VectorXd desiredX = y.row(0).transpose(); // kinematics.getEndEffectorPosition(startJoint);
				VectorXd desiredV = VectorXd::Zero(3);

				//currentAngles = getCurrentAnglesWithImpedanceAvoidance(currentAngles, currentV, desiredX, desiredV, timeStep, currentCartCorrectiveVel);
				getCurrentAnglesWithImpedanceAvoidance(currentAngles, currentV, desiredX, desiredV, timeStep, false, currentCartCorrectiveVel, currentAngleVelocities);

				publishMessagesImpedance(currentAngles, currentAngleVelocities, pub_positions, pub_torque, pub_gains);
				ros::spinOnce();				//need if we have a subscribtion in the program
				loop_rate.sleep();

			}
			//cout << "finish waiting. prepare to start a new one" << endl;
			currentAngles = startJoint;								
			publishMessagesImpedance(currentAngles, currentAngleVelocities, pub_positions, pub_torque, pub_gains);		// correct the starting configuration of the robot beforing starting a new one
			ros::spinOnce();				//need if we have a subscribtion in the program
			loop_rate.sleep();

			s[a]++; 	//keep track of which samples we executed
			// counter++;
    	}


		for(int i = 0; i < nGoals; i++) {	//print and save the weight values for each update
			cout << "weights for goal " << i << endl;
			VectorXd weights = dmpSolver[i]->getCentersWidthsAndWeights();
			cout << weights << endl;
			writeToFileWeights(subjectNr, experimentNr, iUpdate, i+1, weights);
		}

    	for(int i = 0; i < costPerSample.rows(); i++) {		//print total cost for each sample after every update
			cout << "costPerSample: " << costPerSample(i,0) << ", " << flush;		//index has to be corrected
			meanCost += costPerSample(i,0);
    	}
    	meanCost /= costPerSample.rows();
    	cout << "\n" << endl;

		if(iUpdate == 3 || iUpdate == 6) {
			cout << "Time for a break!" << endl;
			cin.ignore();			//pause the experiment to give time for a questionnaire
		}
	}		//end update loop
	cout << "You are done!" << endl;
	return meanCost;
}*/
