//
// Created by dmitr on 16.07.2017.
//

#include "KDTree.h"

#include <iostream>
#include "nanoflann.hpp"

using namespace nanoflann;
using namespace std;

KDTree::KDTree(std::vector<Map::single_landmark_s> landmarks) {

    for(int i = 0; i < landmarks.size(); i++){
        PointCloud<double>::Point pt = PointCloud<double>::Point();
        pt.x = landmarks[i].x_f;
        pt.y = landmarks[i].y_f;
        cloud.pts.push_back(pt);
    }

    index_ = new my_kd_tree_t(2, cloud, KDTreeSingleIndexAdaptorParams(1 /* max leaf */) );
    index_ -> buildIndex();

}

KDTree::PointCloud<double>::Point KDTree::FindClosestPoint(const double x, const double y) {

    double query_pt[2] = { x, y};

    const size_t num_results = 1;
    size_t ret_index;
    double out_dist_sqr;
    nanoflann::KNNResultSet<double> resultSet(num_results);
    resultSet.init(&ret_index, &out_dist_sqr );

    index_ -> findNeighbors(resultSet, &query_pt[0], nanoflann::SearchParams(10));

    return cloud.pts[ret_index];
}

KDTree::~KDTree() {
    delete index_;
}
