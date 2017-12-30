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
#include <limits>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 100;
	default_random_engine gen;

	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);
	for (int i = 0; i < num_particles; i++) {
		Particle p;
		p.id = i;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.weight = 1.0;

		particles.push_back(p);
		weights.push_back(1.0);
	}
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	// add random noise to velocity and yaw rate
	default_random_engine gen;

	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_theta(0, std_pos[2]);

	double yd_dt = yaw_rate*delta_t;
	for (int i = 0; i < num_particles; i++) {
		Particle p = particles[i];
		if (fabs(yaw_rate) > 0.0001) {
			particles[i].x = p.x + velocity * (sin(p.theta + yd_dt) - sin(p.theta))/ yaw_rate;
			particles[i].y = p.y + velocity * (cos(p.theta) - cos(p.theta + yd_dt))/ yaw_rate;
			particles[i].theta = p.theta + yd_dt;
		} else {
			particles[i].x = p.x + velocity * cos(p.theta) * delta_t;
			particles[i].y = p.y + velocity * sin(p.theta) * delta_t;
			particles[i].theta = p.theta;
		}
		particles[i].x += dist_x(gen);
		particles[i].y += dist_y(gen);
		particles[i].theta += dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	//cout << "dataAssociation" << endl;

	for (int i = 0; i < observations.size(); i++) {
		int closest = 0;
		double dist = numeric_limits<double>::max() ;
		int j = 0;
		double o_x = observations[i].x;
		double o_y = observations[i].y;
		do {
			double currDist = //dist(o_x, o_y, predicted[j].x, predicted[j].y);
			sqrt(pow(o_x-predicted[j].x,2) + pow(o_y-predicted[j].y,2));
			if (currDist < dist) {
				dist = currDist;
				closest = j;
			}
			j++;
		} while (j < predicted.size());
		observations[i].id = predicted[closest].id;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	//cout << "updateWeights" << endl;

	double P_a = 1.0 / (2.0 * M_PI * std_landmark[0] * std_landmark[1]);
	//cout << P_a << endl;

	double P_x_denom = 2.0 * std_landmark[0] * std_landmark[0];
	double P_y_denom = 2.0 * std_landmark[1] * std_landmark[1];

	for (int i = 0; i < num_particles; i++) {
		Particle p = particles[i];
		double st = sin(p.theta);
		double ct = cos(p.theta);
		double weight = 1.0;
		for (int j = 0; j < observations.size(); j++) {
			// transform observation into map/particle frame
			LandmarkObs o = observations[j];
			double tObsX = ct * o.x - st * o.y + p.x;
			double tObsY = st * o.x + ct * o.y + p.y;
			// find closest landmark and the distance to it
			int closest = -1;
			double dist = numeric_limits<double>::max() ;
			for (int k = 0; k < map_landmarks.landmark_list.size(); k++) {
				Map::single_landmark_s l = map_landmarks.landmark_list[k];
				double currDist = //dist(tObsX, tObsY, l.x_f, l.y_f);
					sqrt(pow(tObsX-l.x_f, 2) + (tObsY-l.y_f, 2));
				if (currDist < dist) {
					dist = currDist;
					closest = k;
				}
			}
			if (closest == -1)	// no landmark (?)
				continue;
			//cout << closest << " " << dist << endl;
			// calculate multi-variate gaussian distribution
			Map::single_landmark_s l = map_landmarks.landmark_list[closest];
			double P_b = exp(-(pow(tObsX-l.x_f,2)/P_x_denom + pow(tObsY-l.y_f,2)/P_y_denom));
			double prop = P_a * P_b;
			//cout << "prop " << prop << " P_b " << P_b << " " << l.x_f << " " << l.y_f << endl;
			// multiply into current particle's final weight
			weight *= prop;
		}
		particles[i].weight = weight;
		weights[i] = particles[i].weight;
		//cout << weights[i] << endl;

	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	//cout << "resample" << endl;

	default_random_engine gen;
	discrete_distribution<> d(weights.begin(), weights.end());

	vector<Particle> new_particles;
	for (int i = 0; i < num_particles; i++) {
		new_particles.push_back(particles[d(gen)]);
	}
	particles.clear();
	particles = new_particles;
	//cout << "resample end" << endl;

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
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
