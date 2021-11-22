#include <OpenMesh/Core/IO/Options.hh>
#include <OpenMesh/Core/IO/MeshIO.hh>
#include <Eigen/Core>

#include "decimate.h"
#include <iostream>
#include <limits>
#include <queue>


using namespace OpenMesh;
using namespace Eigen;

// NOTE: We do not use pointers in the queue to avoid any memory leak issues.
// When it is guaranteed that everyone can use c+11 functionality,
// we can consider to use std::unique_ptr<> or std::shared_ptr<> later.
typedef std::priority_queue<VertexPriority, std::vector<VertexPriority>,
	VertexPriorityCompare> VertexPriorityQueue;
VPropHandleT<Eigen::Matrix4d> vprop_quadric;
VPropHandleT<int> vprop_latest_version;


// Mesh property accessors
Eigen::Matrix4d& vertex_quadric(Mesh& _mesh, const Mesh::VertexHandle _vh) {
	return _mesh.property(vprop_quadric, _vh);
}

// The vertex version is used for tracking the latest out-going halfedge with
// the minimum priority
int& vertex_latest_version(Mesh& _mesh, const Mesh::VertexHandle _vh) {
	return _mesh.property(vprop_latest_version, _vh);
}


// Functions
void initialize(Mesh& _mesh);
double compute_priority(Mesh& _mesh, const Mesh::HalfedgeHandle _heh);
bool is_collapse_valid(Mesh& _mesh, const Mesh::HalfedgeHandle _heh);
bool is_vertex_priority_valid(Mesh& _mesh, const VertexPriority& _vp);
void enqueue_vertex(Mesh& _mesh, VertexPriorityQueue& _queue, const Mesh::VertexHandle _vh);
void decimate(Mesh& _mesh, const unsigned int _target_num_vertices);


void simplify(Mesh& _mesh, const std::string _output_filename, const float _ratio) {
	// Add required properties
	_mesh.request_vertex_status();
	_mesh.request_edge_status();
	_mesh.request_face_status();
	_mesh.request_face_normals();
	_mesh.add_property(vprop_quadric);
	_mesh.add_property(vprop_latest_version);

	// Compute normals and quadrics
	initialize(_mesh);

	// Decimate
	decimate(_mesh, (int)(_ratio * _mesh.n_vertices()));
	std::cout << "Simplified to #vertices: " << _mesh.n_vertices() << std::endl;

	_mesh.remove_property(vprop_quadric);
	_mesh.remove_property(vprop_latest_version);

	// Write to file
	IO::Options opt;
	std::cout << "Writing to file '" << _output_filename << "'... ";
	if (!IO::write_mesh(_mesh, _output_filename, opt)) {
		std::cout << "Failed!" << std::endl;
	}
	std::cout << "Done." << std::endl;
}

void initialize(Mesh& _mesh) {
	// Compute face normals
	_mesh.update_face_normals();

	for (Mesh::ConstVertexIter v_it = _mesh.vertices_begin();
		v_it != _mesh.vertices_end(); ++v_it) {
		const Mesh::VertexHandle vh = (*v_it);
		vertex_quadric(_mesh, vh).setZero();
		vertex_latest_version(_mesh, vh) = 0;

		// INSERT CODE HERE FOR PART 1-------------------------------------------------------------------------------
		// Calculate vertex quadrics from incident triangles
		// ----------------------------------------------------------------------------------------------------------
		Vec3f point_p = _mesh.point(vh);
		Vector4d coor_p(point_p[0], point_p[1], point_p[2], 1.); // get the coordinates of the point
		Matrix4d qud_q = Matrix4d::Zero();
		for (Mesh::VertexFaceIter vq_it = _mesh.vf_begin(*v_it); vq_it.is_valid(); ++vq_it)
		{
			Vec3f normal = _mesh.normal(*vq_it);
			double a = normal[0];
			double b = normal[1];
			double c = normal[2];
			double norm = sqrt(a*a + b * b + c * c);

			a /= norm;
			b /= norm;
			c /= norm;
			double d = -(a*coor_p[0] + b * coor_p[1] + c * coor_p[2]);
			Vector4d q(a, b, c, d);
			qud_q += q * q.transpose();
		}
		//double q_qud = coor_p.transpose()*qud_q*coor_p;
		vertex_quadric(_mesh, vh) = qud_q;// associate the symmetric matrix Q with the vertex.
	}

	std::cout << "Finished initialization." << std::endl;
}

double compute_priority(Mesh& _mesh, const Mesh::HalfedgeHandle _heh) {
	double priority = 0.0;

	// INSERT CODE HERE FOR PART 2---------------------------------------------------------------------------------
	// Return priority: The smaller the better
	// Use quadrics to estimate approximation error
	// -------------------------------------------------------------------------------------------------------------

	// we need to calculate the bidirectional connection (q1+q2) between _mesh and _heh
	Mesh::VertexHandle i_to_j = _mesh.from_vertex_handle(_heh);
	Mesh::VertexHandle j_to_i = _mesh.to_vertex_handle(_heh);

	Matrix4d& p_i = vertex_quadric(_mesh, i_to_j);
	p_i += vertex_quadric(_mesh, j_to_i);// 

	Vec3f point_i = _mesh.point(j_to_i);
	Vector4d coor_p(point_i[0], point_i[1], point_i[2], 1.0);
	priority = coor_p.transpose()*p_i*coor_p;
	return priority;
}

bool is_collapse_valid(Mesh& _mesh, const Mesh::HalfedgeHandle _heh) {
	const Mesh::VertexHandle from_vh = _mesh.from_vertex_handle(_heh);
	const Mesh::VertexHandle to_vh = _mesh.to_vertex_handle(_heh);

	// Collect faces
	const Mesh::FaceHandle fh_0 = _mesh.face_handle(_heh);
	const Mesh::FaceHandle fh_1 = _mesh.face_handle(_mesh.opposite_halfedge_handle(_heh));

	// Backup point positions
	const Mesh::Point from_p = _mesh.point(from_vh);
	const Mesh::Point to_p = _mesh.point(to_vh);

	// Topological test
	if (!_mesh.is_collapse_ok(_heh))
		return false;

	// Test boundary
	if (_mesh.is_boundary(from_vh) && !_mesh.is_boundary(to_vh))
		return false;

	// Test for normal flipping
	for (Mesh::ConstVertexFaceIter vf_it = _mesh.cvf_begin(from_vh); vf_it != _mesh.cvf_end(from_vh); ++vf_it) {
		const Mesh::FaceHandle n_fh = (*vf_it);
		if (fh_0 == n_fh || fh_1 == n_fh) continue;
		const Mesh::Normal n_before = _mesh.normal(n_fh).normalized();

		Vec3f nf_p[3];
		Mesh::ConstFaceVertexIter n_fv_it = _mesh.cfv_begin(n_fh);
		for (int i = 0; n_fv_it != _mesh.cfv_end(n_fh) && i < 3; ++n_fv_it, ++i) {
			const Mesh::VertexHandle nn_vh = (*n_fv_it);
			nf_p[i] = _mesh.point(nn_vh);

			// Replace 'from' point to 'to' point.
			if (nf_p[0] == from_p) nf_p[0] = to_p;
		}

		const Mesh::Normal cross_prod = cross(nf_p[1] - nf_p[0], nf_p[2] - nf_p[0]);

		if (std::abs(cross_prod.norm()) > 1.0E-8) {
			const Mesh::Normal n_after = cross_prod.normalized();

			// Consider the triangle is flipped if the normal angle is changed more than 45 degrees
			const Mesh::Scalar cos_pi_over_4 = 1 / sqrt(2.0);
			if (dot(n_before, n_after) < cos_pi_over_4)
				return false;
		}
	}

	// Collapse passed all tests
	return true;
}

bool is_vertex_priority_valid(Mesh& _mesh, const VertexPriority& _vp) {
	// The halfedge priority is valid only when its version is equal to the
	// 'from' vertex version.
	return (_vp.version_ == vertex_latest_version(_mesh, _vp.vh_));
}

void enqueue_vertex(Mesh& _mesh, VertexPriorityQueue& _queue,
	const Mesh::VertexHandle _vh) {
	double min_priority = std::numeric_limits<double>::max();
	Mesh::HalfedgeHandle min_heh;

	// Find the minimum priority out-going halfedge
	for (Mesh::ConstVertexOHalfedgeIter vh_it = _mesh.cvoh_begin(_vh); vh_it != _mesh.cvoh_end(_vh); ++vh_it) {
		if (is_collapse_valid(_mesh, *vh_it)) {
			const double priority = compute_priority(_mesh, *vh_it);
			if (priority < min_priority) {
				min_priority = priority;
				min_heh = (*vh_it);
			}
		}
	}

	// Update queue
	if (min_priority < std::numeric_limits<double>::max()) {
		// Increase the vertex version and use the updated version for the halfedge
		int& version = vertex_latest_version(_mesh, _vh);
		++version;
		_queue.emplace(_vh, min_heh, min_priority, version);
	}
}

void decimate(Mesh& _mesh, const unsigned int _target_num_vertices) {
	std::cout << "Starting decimation... ";

	// Build priority queue
	VertexPriorityQueue queue;
	for (Mesh::ConstVertexIter v_it = _mesh.vertices_begin();
		v_it != _mesh.vertices_end(); ++v_it) {
		const Mesh::VertexHandle vh = (*v_it);
		enqueue_vertex(_mesh, queue, vh);
	}

	int num_vertices = _mesh.n_vertices();
	// INSERT CODE HERE FOR PART 3-----------------------------------------------------------------------------------
	// Decimate using priority queue:
	//
	// 1) Take first element of queue
	//  - Check whether the vertex priority is valid using 'is_vertex_priority_valid()'
	//
	// 2) Collapse this halfedge
	//  - Check whether the halfedge collapse is valid using 'is_collapse_valid()'
	//
	// 3) Update queue
	//
	// --------------------------------------------------------------------------------------------------------------
	int num_val_vet = num_vertices - _target_num_vertices;
	for (int i = 0; i < num_val_vet && !(queue.empty()); i++)
	{

		const VertexPriority *start = &queue.top();
		// get the first element of the queue that has the valide priority and collapse.
		while (!(is_collapse_valid(_mesh, (*start).heh_)) || !(is_vertex_priority_valid(_mesh, *start)))
		{
			queue.pop();
			start = &queue.top();
		}
		VertexHandle s_point = start->vh_;
		VertexHandle e_point = _mesh.to_vertex_handle(start->heh_);

		Matrix4d q = vertex_quadric(_mesh, s_point);
		vertex_quadric(_mesh, e_point) += q;


		_mesh.collapse(start->heh_);

		// return the updated vertice set
		std::vector<Mesh::VertexHandle> updated;
		for (Mesh::VertexVertexIter it = _mesh.vv_iter(e_point); it.is_valid(); ++it)
		{
			updated.push_back(*it);
		}
		updated.push_back(e_point);

		for (Mesh::VertexHandle& item : updated)
		{
			enqueue_vertex(_mesh, queue, item);
		}
	}
	queue.empty();
	// Delete the items marked to be deleted
	_mesh.garbage_collection();
	std::cout << "Done." << std::endl;
}
