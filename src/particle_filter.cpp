/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <array>

#include "particle_filter.h"
#include "KDTree.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {

    // Number of particles, for 1000 particles execution around 63 sec
    num_particles = 100;
    // Resize weight vector for stor
    weights.resize(num_particles);

    // Setup random generators for x, y and theta
    default_random_engine gen;
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);

    // Create initial particles
    for(int i = 0; i < num_particles; i++){
        Particle particle = Particle();
        particle.x = dist_x(gen);
        particle.y = dist_y(gen);
        particle.theta = dist_theta(gen);
        particle.weight = 1;

        particles.push_back(particle);
    }

    is_initialized = true;

    //KdTree would be used for nearest neigboor search
    use_kd_tree = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {

    normal_distribution<double> dist_x(0, std_pos[0]);
    normal_distribution<double> dist_y(0, std_pos[1]);
    normal_distribution<double> dist_theta(0, std_pos[2]);

    default_random_engine gen;

    //Predict particles position based on measurements;
    for(auto&& particle : particles){
        if(fabs(yaw_rate) > 0.0001){
            particle.x += velocity/yaw_rate*(sin(particle.theta + yaw_rate*delta_t) - sin(particle.theta)) + dist_x(gen);
            particle.y += velocity/yaw_rate*(cos(particle.theta) - cos(particle.theta + yaw_rate*delta_t)) + dist_y(gen);
            particle.theta += yaw_rate*delta_t + dist_theta(gen);
        } else {
            particle.x += velocity*cos(particle.theta)*delta_t + dist_x(gen);
            particle.y += velocity*sin(particle.theta)*delta_t + dist_y(gen);
            particle.theta += dist_theta(gen);
        }
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {

    // Init KDTree with landmarks;
    KDTree tree(map_landmarks.landmark_list);

    // Preinit some numbers for later weight calculations
    double var_x = pow(std_landmark[0], 2);
    double var_y = pow(std_landmark[1], 2);
    double covar_xy = std_landmark[0] * std_landmark[1];
    double weights_sum = 0;

    for(int i = 0; i < particles.size(); i++){

        Particle& particle = particles[i];

        double weight = 1.0;

        for(int j = 0; j < observations.size(); j++){
           // Transform observation coordinats to Map space
            double predicted_x = observations[j].x*cos(particle.theta) - observations[j].y*sin(particle.theta) + particle.x;
            double predicted_y = observations[j].x*sin(particle.theta) + observations[j].y*cos(particle.theta) + particle.y;

            double x_diff = 0;
            double y_diff = 0;

            if(use_kd_tree){
                auto&& point = tree.FindClosestPoint(predicted_x, predicted_y);

                x_diff = predicted_x - point.x;
                y_diff = predicted_y - point.y;
            } else {
                Map::single_landmark_s nearest_landmark = findClosestLandmark(sensor_range, map_landmarks, predicted_x,
                                                                              predicted_y);
                x_diff = predicted_x - nearest_landmark.x_f;
                y_diff = predicted_y - nearest_landmark.y_f;
            }


            double numer = exp(-0.5*((x_diff * x_diff)/var_x + (y_diff * y_diff)/var_y));
            double denom = 2 * M_PI*covar_xy;
            // multiply particle weight by this obs-weight pair stat
            weight *= numer/denom;
        }

        // Update weights
        particle.weight = weight;
        weights[i] = weight;
        weights_sum += weight;
    }

    normalizeWeights(weights_sum);
}

Map::single_landmark_s ParticleFilter::findClosestLandmark(double sensor_range, const Map &map_landmarks,
                                                           double predicted_x, double predicted_y) {
    double min_distance = sensor_range;
    int landmark_min_idx = -1;
    for(int k = 0; k < map_landmarks.landmark_list.size(); k++){
                    Map::single_landmark_s landmark = map_landmarks.landmark_list[k];

                    double distance = sqrt(pow(predicted_x - landmark.x_f, 2) + pow(predicted_y - landmark.y_f, 2));

                    if(distance < min_distance){
                        min_distance = distance;
                        landmark_min_idx = k;
                    }
                }

    Map::single_landmark_s nearest_landmark = map_landmarks.landmark_list[landmark_min_idx];
    return nearest_landmark;
}

void ParticleFilter::normalizeWeights(const double weights_sum) {
    for(int i = 0; i < num_particles; i++){
        weights[i] = weights[i] / weights_sum;
    }
}

void ParticleFilter::resample() {
    default_random_engine gen;
    std::discrete_distribution<> d(weights.begin(), weights.end());

    // Resample particles with replacement with probability proportional to their weight.
    std::vector<Particle> updated_particles;
    for(int i = 0; i < particles.size(); i++){
        int particle_id = d(gen);
        updated_particles.push_back(particles[particle_id]);
    }

    particles = updated_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
