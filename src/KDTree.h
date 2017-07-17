//
// Created by dmitr on 16.07.2017.
//

#ifndef PARTICLE_FILTER_KDTREE_H
#define PARTICLE_FILTER_KDTREE_H


#include <vector>
#include "map.h"
#include "nanoflann.hpp"
#include "helper_functions.h"

using namespace nanoflann;

class KDTree {

public:
    template <typename T>
    struct PointCloud
    {
        struct Point
        {
            T  x,y;
        };

        std::vector<Point>  pts;

        // Must return the number of data points
        inline size_t kdtree_get_point_count() const { return pts.size(); }

        // Returns the distance between the vector "p1[0:size-1]" and the data point with index "idx_p2" stored in the class:
        inline T kdtree_distance(const T *p1, const size_t idx_p2,size_t /*size*/) const
        {
            const T d0=p1[0]-pts[idx_p2].x;
            const T d1=p1[1]-pts[idx_p2].y;
            return d0*d0+d1*d1;
        }

        // Returns the dim'th component of the idx'th point in the class:
        // Since this is inlined and the "dim" argument is typically an immediate value, the
        //  "if/else's" are actually solved at compile time.
        inline T kdtree_get_pt(const size_t idx, int dim) const
        {
            if (dim==0) return pts[idx].x;
            else return pts[idx].y;
        }

        // Optional bounding-box computation: return false to default to a standard bbox computation loop.
        //   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
        //   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
        template <class BBOX>
        bool kdtree_get_bbox(BBOX& /*bb*/) const { return false; }

    };

    KDTree(std::vector<Map::single_landmark_s> landmarks);

    PointCloud<double>::Point FindClosestPoint(const double x, const double y);

    typedef KDTreeSingleIndexAdaptor<L2_Simple_Adaptor<double, PointCloud<double>>, PointCloud<double>, 2> my_kd_tree_t;
    PointCloud<double> cloud;
    my_kd_tree_t* index_;

    virtual ~KDTree();


};


#endif //PARTICLE_FILTER_KDTREE_H
