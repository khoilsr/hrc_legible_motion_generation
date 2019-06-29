/**
 * @file runEvolutionaryOptimization.cpp
 * @brief  Source file for function to run an evolutionary optimization process.
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

#include "/home/hrcstudents/Documents/Code/catkin_ws/src/dmp_bbo_node/include/dmp_bbo_node/runEvolutionaryOptimization.hpp"

#include <iomanip>
#include <fstream>
#include <boost/filesystem.hpp>
#include <eigen3/Eigen/Core>

#include "/home/hrcstudents/Documents/Code/catkin_ws/src/dmp_bbo_node/include/dmp_bbo_node/DistributionGaussian.hpp"
#include "/home/hrcstudents/Documents/Code/catkin_ws/src/dmp_bbo_node/include/dmp_bbo_node/Updater.hpp"
#include "/home/hrcstudents/Documents/Code/catkin_ws/src/dmp_bbo_node/include/dmp_bbo_node/UpdateSummary.hpp"
//#include "/home/hrcstudents/Documents/Code/catkin_ws/src/dmp_bbo_node/include/dmp_bbo_node/CostFunction.hpp"
#include "/home/hrcstudents/Documents/Code/catkin_ws/src/dmp_bbo_node/include/dmp_bbo_node/Task.hpp"
#include "/home/hrcstudents/Documents/Code/catkin_ws/src/dmp_bbo_node/include/dmp_bbo_node/TaskSolver.hpp"
//#include "/home/hrcstudents/Documents/Code/catkin_ws/src/dmp_bbo_node/include/dmp_bbo_node/ExperimentBBO.hpp"

#include "/home/hrcstudents/Documents/Code/catkin_ws/src/dmp_bbo_node/include/dmp_bbo_node/EigenFileIO.hpp"




using namespace std;
using namespace Eigen;

namespace DmpBbo {

/*void runEvolutionaryOptimization(
  const CostFunction* const cost_function,
  const DistributionGaussian* const initial_distribution,
  const Updater* const updater,
  int n_updates,
  int n_samples_per_update,
  std::string save_directory,
  bool overwrite,
  bool only_learning_curve)
{

  // Some variables
  MatrixXd samples;
  VectorXd costs, cost_eval;
  // Bookkeeping
  vector<UpdateSummary> update_summaries;
  UpdateSummary update_summary;

  if (save_directory.empty())
    cout << "init  =  " << "  distribution=" << *initial_distribution;

  // Optimization loop
  DistributionGaussian* distribution = initial_distribution->clone();
  for (int i_update=0; i_update<n_updates; i_update++)
  {
    // 0. Get cost of current distribution mean
    cost_function->evaluate(distribution->mean().transpose(),cost_eval);

    // 1. Sample from distribution
    distribution->generateSamples(n_samples_per_update, samples);

    // 2. Evaluate the samples
    cost_function->evaluate(samples,costs);

    // 3. Update parameters
    updater->updateDistribution(*distribution, samples, costs, *distribution, update_summary);


    // Some output and/or saving to file (if "directory" is set)
    if (save_directory.empty())
    {
      cout << "\t cost_eval=" << cost_eval << endl << i_update+1 << "  " << *distribution;
    }
    else
    {
      update_summary.cost_eval = cost_eval[0];
      update_summaries.push_back(update_summary);
    }
  }

  // Save update summaries to file, if necessary
  if (!save_directory.empty())
    saveToDirectory(update_summaries,save_directory,overwrite,only_learning_curve);

}*/

// This function could have been integrated with the above. But I preferred to duplicate a bit of
// code so that the difference between running an optimziation with a CostFunction or
// Task/TaskSolver is more apparent.
void runEvolutionaryOptimization(
  const Task* const task,
  const TaskSolver* const task_solver,
  const DistributionGaussian* const initial_distribution,
  const Updater* const updater,
  int n_updates,
  int n_samples_per_update,
  std::string save_directory,
  bool overwrite,
  bool only_learning_curve)
{
  // Some variables
    MatrixXd samples;
    MatrixXd cost_vars, cost_vars_eval;
    VectorXd costs, cost_eval;
  // Bookkeeping
  vector<UpdateSummary> update_summaries;
  UpdateSummary update_summary;

  if (save_directory.empty())
    cout << "init  =  " << "  distribution=" << *initial_distribution;

  // Optimization loop
  DistributionGaussian* distribution = initial_distribution->clone();
  for (int i_update=0; i_update<n_updates; i_update++)
  {
    // 0. Get cost of current distribution mean
    task_solver->performRollouts(distribution->mean().transpose(),cost_vars_eval);
    task->evaluate(cost_vars_eval,cost_eval);

    // 1. Sample from distribution
    distribution->generateSamples(n_samples_per_update, samples);




    // 2A. Perform the roll-outs
    task_solver->performRollouts(samples,cost_vars);

    // 2B. Evaluate the samples
    task->evaluate(cost_vars,costs);

    // 3. Update parameters
    updater->updateDistribution(*distribution, samples, costs, *distribution, update_summary);


    // Some output and/or saving to file (if "directory" is set)
    if (save_directory.empty())
    {
      cout << "\t cost_eval=" << cost_eval << endl << i_update+1 << "  " << *distribution;
    }
    else
    {
      update_summary.cost_eval = cost_eval[0];
      update_summary.cost_vars_eval = cost_vars_eval;
      update_summary.cost_vars = cost_vars;
      update_summaries.push_back(update_summary);
    }
  }

  // Save update summaries to file, if necessary
  if (!save_directory.empty())
  {
    saveToDirectory(update_summaries,save_directory,overwrite,only_learning_curve);
    // If you store only the learning curve, no need to save the script to visualize rollouts
    if (!only_learning_curve)
      task->savePerformRolloutsPlotScript(save_directory);
  }

}

void runEvolutionaryOptimizationHrc(
  const Task* const task,
  const TaskSolver* const task_solver1, const TaskSolver* const task_solver2,
  const DistributionGaussian* const initial_distribution1,   const DistributionGaussian* const initial_distribution2,
  const Updater* const updater1,
          const Updater* const updater2,
  int n_updates,
  int n_samples_per_update,
  std::string save_directory,
  bool overwrite,
  bool only_learning_curve)
{
  // Some variables


  MatrixXd samples1;
    MatrixXd samples2;
  MatrixXd cost_vars1, cost_vars_eval1;
  MatrixXd cost_vars2, cost_vars_eval2;
  VectorXd costs1, cost_eval1;
  VectorXd costs2, cost_eval2;
  // Bookkeeping
  vector<UpdateSummary> update_summaries1;
    vector<UpdateSummary> update_summaries2;
  UpdateSummary update_summary1;
  UpdateSummary update_summary2;

  if (save_directory.empty()){
    cout << "init  =  " << "  distribution 1 =" << *initial_distribution1;
   cout << "init  =  " << "  distribution 2 =" << *initial_distribution2;}

  // Optimization loop
  DistributionGaussian* distribution1 = initial_distribution1->clone();
   DistributionGaussian* distribution2 = initial_distribution2->clone();
  for (int i_update=0; i_update<n_updates; i_update++)
  {
    // 0. Get cost of current distribution mean
    task_solver1->performRollouts(distribution1->mean().transpose(),cost_vars_eval1);
task_solver2->performRollouts(distribution2->mean().transpose(),cost_vars_eval2);
    cout << "Update no. =  " << i_update <<endl;

    task->evaluateHrc(cost_vars_eval1,cost_vars_eval2,cost_eval1,cost_eval2,(i_update+1)*100);

    // 1. Sample from distribution
    distribution1->generateSamples(n_samples_per_update, samples1);
        distribution2->generateSamples(n_samples_per_update, samples2);


    Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
    std::cout <<"samples 1 weights for update no. " <<i_update<< samples1.format(CleanFmt) << endl;
        std::cout <<"samples 2 weights for update no. " <<i_update<< samples2.format(CleanFmt) << endl;

    // 2A. Perform the roll-outs
    task_solver1->performRollouts(samples1,cost_vars1);
  task_solver2->performRollouts(samples2,cost_vars2);

    // 2B. Evaluate the samples
    task->evaluateHrc(cost_vars1,cost_vars2,costs1,costs2,i_update);

    // 3. Update parameters
    updater1->updateDistribution(*distribution1, samples1, costs1, *distribution1, update_summary1);
    updater2->updateDistribution(*distribution2, samples2, costs2, *distribution2, update_summary2);


    // Some output and/or saving to file (if "directory" is set)
    if (save_directory.empty())
    {
      cout << "\t cost_eval1 =" << cost_eval1 << endl << i_update+1 << "  " << *distribution1;
       cout << "\t cost_eval2 =" << cost_eval2 << endl << i_update+1 << "  " << *distribution2;
    }
    else
    {
      update_summary1.cost_eval = cost_eval1[0];
      update_summary1.cost_vars_eval = cost_vars_eval1;
      update_summary1.cost_vars = cost_vars1;
      update_summaries1.push_back(update_summary1);

      update_summary2.cost_eval = cost_eval2[0];
      update_summary2.cost_vars_eval = cost_vars_eval2;
      update_summary2.cost_vars = cost_vars2;
      update_summaries2.push_back(update_summary2);
    }
  }

  // Save update summaries to file, if necessary
  if (!save_directory.empty())
  {
      std::string string1= "/Goal1";
      std::string string2= "/Goal2";
    saveToDirectory(update_summaries1,save_directory+string1,overwrite,only_learning_curve);
    saveToDirectory(update_summaries2,save_directory+string2,overwrite,only_learning_curve);
    // If you store only the learning curve, no need to save the script to visualize rollouts
    if (!only_learning_curve)
      task->savePerformRolloutsPlotScriptHrc(save_directory);
  }

}

/*void runEvolutionaryOptimization(ExperimentBBO* experiment, std::string save_directory, bool overwrite,   bool only_learning_curve)
{
 if (experiment->cost_function!=NULL)
 {
   runEvolutionaryOptimization(
     experiment->cost_function,
     experiment->initial_distribution,
     experiment->updater,
     experiment->n_updates,
     experiment->n_samples_per_update,
     save_directory,
     overwrite,
     only_learning_curve);
 }
 else
 {
   runEvolutionaryOptimization(
     experiment->task,
     experiment->task_solver,
     experiment->initial_distribution,
     experiment->updater,
     experiment->n_updates,
     experiment->n_samples_per_update,
     save_directory,
     overwrite,
     only_learning_curve);
 }
}*/


}
