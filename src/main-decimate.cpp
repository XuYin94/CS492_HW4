#include <OpenMesh/Core/IO/Options.hh>
#include <OpenMesh/Core/IO/MeshIO.hh>
#include <iostream>
#include <cmath>
#include <cstdio>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include "decimate.h"

using namespace std;
using namespace OpenMesh;
using namespace Eigen;

int main(int argc, char** argv) {
  if (argc < 4) {
    cout << "Usage: " << argv[0] << " in_mesh_filename out_mesh_filename ratio\n";
    exit(0);
  }

  IO::Options opt;
  opt += IO::Options::VertexNormal;
  opt += IO::Options::FaceNormal;

  Mesh mesh;
  mesh.request_face_normals();
  mesh.request_vertex_normals();

  cout << "Reading from file " << argv[1] << "...\n";
  if (!IO::read_mesh(mesh, argv[1], opt)) {
    cout << "Read failed.\n";
    exit(0);
  }

  cout << "Mesh stats:\n";
  cout << '\t' << mesh.n_vertices() << " vertices.\n";
  cout << '\t' << mesh.n_edges() << " edges.\n";
  cout << '\t' << mesh.n_faces() << " faces.\n";

  mesh.update_normals();

  const float ratio = std::stof(argv[3]);
  if (ratio <= 0.0f || ratio >= 1.0f) {
    std::cout << "Failed: The ratio must be in (0, 1) range." << std::endl;
    std::cout << "Ratio: " << ratio << std::endl;
    return -1;
  }

  simplify(mesh, argv[2], ratio);

  return 0;
}
