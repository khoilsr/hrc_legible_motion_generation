/*beim ersten Aufruf cost_vars_eval eingeben, beim zweiten cost_vars (MatrixXd)
  beim ersten Aufruf cost_eval eingeben, beim 2. costs (VectorXd)
  set first to 1, if it's the fist use of the function, set to 0 if it's the second use */
void performRolloutsAndPublishMessages(MatrixXd cost_vars1, MatrixXd cost_vars2, MatrixXd cost_vars3,/* int first,*/ VectorXd costs1, VectorXd costs2, VectorXd costs3) {
    task_solver1->performRollouts(distribution1->mean().transpose(), cost_vars1);            
    task_solver2->performRollouts(distribution2->mean().transpose(), cost_vars2);
    task_solver3->performRollouts(distribution3->mean().transpose(), cost_vars3);

    /*if(first) {
        cout << "Update no. =  " << i_update << endl;   //ist nicht da im 2.
    }*/

    for (int i = 0; i < 1; i++) {

        int n_samples = cost_vars1.rows();
        int n_dims = goal1.size();
        int n_cost_vars = 4 * n_dims + 1; // y_1..y_D  yd_1..yd_D  ydd_1..ydd_D  t forcing_term_1..forcing_term_D
        int n_time_steps = cost_vars1.cols() / n_cost_vars;

        time_weight = 2.5;

        costs1.resize(n_samples);          
        costs2.resize(n_samples);
        costs3.resize(n_samples);

        MatrixXd rollout; //(n_time_steps,n_cost_vars);
       
        MatrixXd rolloutFK(n_time_steps, 3);
        
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
                my_row = cost_vars1.row(i1);
                rollout = (Map<MatrixXd>(my_row.data(), n_cost_vars, n_time_steps)).transpose();
                yd = rollout.block(0, n_dims, n_time_steps, n_dims);
                ydd = rollout.block(0, 2*n_dims, n_time_steps, n_dims);
                // std::cout << "a is now of size " << yd.rows() << "x" << yd.cols() << std::endl;
                // cout << yd(0, 0) << "\t" << yd(0, 1) << "\t" << yd(0, 2) <<"\t" << yd(0, 3) <<"\t" << yd(0, 4) <<"\t" << yd(0, 5) <<"\t" << yd(0, 6) <<endl;

                ts = rollout.col(3 * n_dims);

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

                    // cout << yd(hh, 0) << "\t" << yd(hh, 1) << "\t" << yd(hh, 2) <<"\t" << yd(hh, 3) <<"\t" << yd(hh, 4) <<"\t" << yd(hh, 5) <<"\t" << yd(hh, 6) <<endl;

                    msg_.points[0].velocities[0] = yd(hh, 0) ;
                    msg_.points[0].velocities[1] = yd(hh, 1) ;
                    msg_.points[0].velocities[6] = yd(hh, 2) ;
                    msg_.points[0].velocities[2] = yd(hh, 3) ;
                    msg_.points[0].velocities[3] = yd(hh, 4) ;
                    msg_.points[0].velocities[4] = yd(hh, 5) ;
                    msg_.points[0].velocities[5] = yd(hh, 6) ;

                    msg_.points[0].accelerations[0] = ydd(hh, 0) ;
                    msg_.points[0].accelerations[1] = ydd(hh, 1) ;
                    msg_.points[0].accelerations[6] = ydd(hh, 2) ;
                    msg_.points[0].accelerations[2] = ydd(hh, 3) ;
                    msg_.points[0].accelerations[3] = ydd(hh, 4) ;
                    msg_.points[0].accelerations[4] = ydd(hh, 5) ;
                    msg_.points[0].accelerations[5] = ydd(hh, 6) ;


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
                    msg_.points[0].velocities[0] = -yd(hh - 1, 0) ;
                    msg_.points[0].velocities[1] = -yd(hh - 1, 1) ;
                    msg_.points[0].velocities[6] = -yd(hh - 1, 2) ;
                    msg_.points[0].velocities[2] = -yd(hh - 1, 3) ;
                    msg_.points[0].velocities[3] = -yd(hh - 1, 4) ;
                    msg_.points[0].velocities[4] = -yd(hh - 1, 5) ;
                    msg_.points[0].velocities[5] = -yd(hh - 1, 6) ;
                    msg_.points[0].accelerations[0] = -ydd(hh - 1, 0) ;
                    msg_.points[0].accelerations[1] = -ydd(hh - 1, 1) ;
                    msg_.points[0].accelerations[6] = -ydd(hh - 1, 2) ;
                    msg_.points[0].accelerations[2] = -ydd(hh - 1, 3) ;
                    msg_.points[0].accelerations[3] = -ydd(hh - 1, 4) ;
                    msg_.points[0].accelerations[4] = -ydd(hh - 1, 5) ;
                    msg_.points[0].accelerations[5] = -ydd(hh - 1, 6) ;

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
                my_row = cost_vars2.row(i2);           //cost_vars1,2,3,  i1,2,3
                rollout = (Map<MatrixXd>(my_row.data(), n_cost_vars, n_time_steps)).transpose();
                yd = rollout.block(0, n_dims, n_time_steps, n_dims);
                ydd = rollout.block(0, 2*n_dims, n_time_steps, n_dims);
                // std::cout << "yd is now of size " << yd.rows() << "x" << yd.cols() << std::endl;
                // cout << yd(0, 0) << "\t" << yd(0, 1) << "\t" << yd(0, 2) <<"\t" << yd(0, 3) <<"\t" << yd(0, 4) <<"\t" << yd(0, 5) <<"\t" << yd(0, 6) <<endl;

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

                cout << "Press enter to start trajectory G2" << endl;       //G1,2,3
                beginTime = ros::Time::now().toSec();

                for (int hh = 0; hh < ts.size(); hh += sampleSize) {
                    msg_.points[0].positions[0] = rollout(hh, 0) ;
                    msg_.points[0].positions[1] = rollout(hh, 1) ;
                    msg_.points[0].positions[6] = rollout(hh, 2) ;
                    msg_.points[0].positions[2] = rollout(hh, 3) ;
                    msg_.points[0].positions[3] = rollout(hh, 4) ;
                    msg_.points[0].positions[4] = rollout(hh, 5) ;
                    msg_.points[0].positions[5] = rollout(hh, 6) ;

                    // cout << yd(hh, 0) << "\t" << yd(hh, 1) << "\t" << yd(hh, 2) <<"\t" << yd(hh, 3) <<"\t" << yd(hh, 4) <<"\t" << yd(hh, 5) <<"\t" << yd(hh, 6) <<endl;   

                    msg_.points[0].velocities[0] = yd(hh, 0) ;
                    msg_.points[0].velocities[1] = yd(hh, 1) ;
                    msg_.points[0].velocities[6] = yd(hh, 2) ;
                    msg_.points[0].velocities[2] = yd(hh, 3) ;
                    msg_.points[0].velocities[3] = yd(hh, 4) ;
                    msg_.points[0].velocities[4] = yd(hh, 5) ;
                    msg_.points[0].velocities[5] = yd(hh, 6) ;

                    msg_.points[0].accelerations[0] = ydd(hh, 0) ;
                    msg_.points[0].accelerations[1] = ydd(hh, 1) ;
                    msg_.points[0].accelerations[6] = ydd(hh, 2) ;
                    msg_.points[0].accelerations[2] = ydd(hh, 3) ;
                    msg_.points[0].accelerations[3] = ydd(hh, 4) ;
                    msg_.points[0].accelerations[4] = ydd(hh, 5) ;
                    msg_.points[0].accelerations[5] = ydd(hh, 6) ;


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
                ss << universalCount2;          //universalCount1,2,3
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
                    msg_.points[0].velocities[0] = -yd(hh - 1, 0) ;
                    msg_.points[0].velocities[1] = -yd(hh - 1, 1) ;
                    msg_.points[0].velocities[6] = -yd(hh - 1, 2) ;
                    msg_.points[0].velocities[2] = -yd(hh - 1, 3) ;
                    msg_.points[0].velocities[3] = -yd(hh - 1, 4) ;
                    msg_.points[0].velocities[4] = -yd(hh - 1, 5) ;
                    msg_.points[0].velocities[5] = -yd(hh - 1, 6) ;
                    msg_.points[0].accelerations[0] = -ydd(hh - 1, 0) ;
                    msg_.points[0].accelerations[1] = -ydd(hh - 1, 1) ;
                    msg_.points[0].accelerations[6] = -ydd(hh - 1, 2) ;
                    msg_.points[0].accelerations[2] = -ydd(hh - 1, 3) ;
                    msg_.points[0].accelerations[3] = -ydd(hh - 1, 4) ;
                    msg_.points[0].accelerations[4] = -ydd(hh - 1, 5) ;
                    msg_.points[0].accelerations[5] = -ydd(hh - 1, 6) ;

                    traj_pub.publish(msg_);
                    ros::spinOnce();
                    loop_rate.sleep();
                }

                if (predictedGoal == "0")           //0 bei a=2, 1 bei a=1, 2 bei a=3
                    accuracy = 0;
                else
                    accuracy = 1;
                cout << "humans time is:" << human_time << endl;
                cout << "accuracy is:" << accuracy << endl;
                computeCosts(costs_detailed2, rollout, rolloutFK, costs2, a, i2, (i_update + 1) * 100, ts, dir2, goal2, n_samples);     //costs_detailed1,2,3, costs1,2,3, i1,2,3, dir1,2,3, goal1,2,3

                i2++;
            }

            if (a == 3 && i3 < n_samples) {

                cout << "Press enter to resume" << endl;
                cin.ignore();
                my_row = cost_vars3.row(i3);           //cost_vars1,2,3, i1,2,3
                rollout = (Map<MatrixXd>(my_row.data(), n_cost_vars, n_time_steps)).transpose();
                yd = rollout.block(0, n_dims, n_time_steps, n_dims);
                ydd = rollout.block(0, 2*n_dims, n_time_steps, n_dims);
                // cout << yd(0, 0) << "\t" << yd(0, 1) << "\t" << yd(0, 2) <<"\t" << yd(0, 3) <<"\t" << yd(0, 4) <<"\t" << yd(0, 5) <<"\t" << yd(0, 6) <<endl;

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

                cout << "Press enter to start trajectory G3" << endl;       //G1,2,3
                beginTime = ros::Time::now().toSec();

                for (int hh = 0; hh < ts.size(); hh += sampleSize) {
                    msg_.points[0].positions[0] = rollout(hh, 0) ;
                    msg_.points[0].positions[1] = rollout(hh, 1) ;
                    msg_.points[0].positions[6] = rollout(hh, 2) ;
                    msg_.points[0].positions[2] = rollout(hh, 3) ;
                    msg_.points[0].positions[3] = rollout(hh, 4) ;
                    msg_.points[0].positions[4] = rollout(hh, 5) ;
                    msg_.points[0].positions[5] = rollout(hh, 6) ;
                    msg_.points[0].velocities[0] = yd(hh, 0) ;
                    msg_.points[0].velocities[1] = yd(hh, 1) ;
                    msg_.points[0].velocities[6] = yd(hh, 2) ;
                    msg_.points[0].velocities[2] = yd(hh, 3) ;
                    msg_.points[0].velocities[3] = yd(hh, 4) ;
                    msg_.points[0].velocities[4] = yd(hh, 5) ;
                    msg_.points[0].velocities[5] = yd(hh, 6) ;
                    msg_.points[0].accelerations[0] = ydd(hh, 0) ;
                    msg_.points[0].accelerations[1] = ydd(hh, 1) ;
                    msg_.points[0].accelerations[6] = ydd(hh, 2) ;
                    msg_.points[0].accelerations[2] = ydd(hh, 3) ;
                    msg_.points[0].accelerations[3] = ydd(hh, 4) ;
                    msg_.points[0].accelerations[4] = ydd(hh, 5) ;
                    msg_.points[0].accelerations[5] = ydd(hh, 6) ;          
                   
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
                ss << universalCount3;          //universalCount1,2,3
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
                    msg_.points[0].velocities[0] = -yd(hh - 1, 0) ;
                    msg_.points[0].velocities[1] = -yd(hh - 1, 1) ;
                    msg_.points[0].velocities[6] = -yd(hh - 1, 2) ;
                    msg_.points[0].velocities[2] = -yd(hh - 1, 3) ;
                    msg_.points[0].velocities[3] = -yd(hh - 1, 4) ;
                    msg_.points[0].velocities[4] = -yd(hh - 1, 5) ;
                    msg_.points[0].velocities[5] = -yd(hh - 1, 6) ;
                    msg_.points[0].accelerations[0] = -ydd(hh - 1, 0) ;
                    msg_.points[0].accelerations[1] = -ydd(hh - 1, 1) ;
                    msg_.points[0].accelerations[6] = -ydd(hh - 1, 2) ;
                    msg_.points[0].accelerations[2] = -ydd(hh - 1, 3) ;
                    msg_.points[0].accelerations[3] = -ydd(hh - 1, 4) ;
                    msg_.points[0].accelerations[4] = -ydd(hh - 1, 5) ;
                    msg_.points[0].accelerations[5] = -ydd(hh - 1, 6) ;

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
                computeCosts(costs_detailed3, rollout, rolloutFK, costs3, a, i3, (i_update + 1) * 100, ts, dir3, goal3, n_samples);         //1,2,3

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

        std::cout << "costs 1 = " << costs1.format(CleanFmt) << endl;          
        std::cout << "costs 2 = " << costs2.format(CleanFmt) << endl;
        std::cout << "costs 3 = " << costs3.format(CleanFmt) << endl;
    }

}

