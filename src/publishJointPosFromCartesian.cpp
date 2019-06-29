#include "/home/hrcstudents/Documents/dmpbbo_ws/src/predictable_kuka_v1/include/Kinematics.hpp"
#include "ros/ros.h"
#include "geometry_msgs/Point.h"
#include <std_msgs/Float64MultiArray.h>

using namespace std;
using namespace Eigen;

Kinematics kinematics;
VectorXd currentAngles(7);


VectorXd integrateVector(VectorXd& prevValue, VectorXd& derivative, double timeStep){
	return prevValue + derivative*timeStep;
}

void computeJointPositions(VectorXd desiredX) {
	VectorXd currentX = kinematics.getEndeffectorPosition(currentAngles);
	VectorXd currentV = VectorXd::Zero(3); 		
	VectorXd desiredV = VectorXd::Zero(3);	//no velocity data received from nadas code
	double timeStep = 0.01;		//not defined, no info about dmp

	VectorXd currentAngleVelocities = kinematics.getCurrentVelocities(desiredX, desiredV, currentAngles, timeStep);
	currentAngles = integrateVector(currentAngles, currentAngleVelocities, timeStep);

	std_msgs::Float64MultiArray msg;
	for(int i = 0; i < 7; i++) {
		msg.data[i] = currentAngles[i];
	}
	pub_positions.publish(msg_positions);
	ros::spinOnce();				//need if we have a subscribtion in the program
	loop_rate.sleep();
}


void cartesianPosreceived(geometry_msgs::Point msg) {
	VectorXd desiredX = VectorXd(msg.x, msg.y, msg.z);

	computeJointPositions(desiredX);
}


int main(int argc, char const *argv[])
{
	
	ros::init(argc, argv, "dmp_cart2joint");
	ros::NodeHandle node;

	VectorXd startJoint(7); startJoint << -160.7 * PI / 180, -26.54 * PI / 180, -156.9* PI / 180, -81.03 * PI / 180, -115.88 * PI / 180, 0, 0;
	currentAngles = startJoint;

	ros::Rate loop_rate(55);
	pub_positions = node.advertise<std_msgs::Float64MultiArray>("/standing2/joint_impedance_controller/position", 100);
	sub_positions = node.subscribe<geometry_msgs::Point>("Standing2/kukaControl/Pos", 100, cartesianPosreceived);




	return 0;
}