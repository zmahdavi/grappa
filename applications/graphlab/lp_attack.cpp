////////////////////////////////////////////////////////////////////////
// This file is part of Grappa, a system for scaling irregular
// applications on commodity clusters. 

// Copyright (C) 2010-2014 University of Washington and Battelle
// Memorial Institute. University of Washington authorizes use of this
// Grappa software.

// Grappa is free software: you can redistribute it and/or modify it
// under the terms of the Affero General Public License as published
// by Affero, Inc., either version 1 of the License, or (at your
// option) any later version.

// Grappa is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// Affero General Public License for more details.

// You should have received a copy of the Affero General Public
// License along with this program. If not, you may obtain one from
// http://www.affero.org/oagl.html.
////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////
/// Use the GraphLab API to implement Connected Components
////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <fstream>
#include <Grappa.hpp>
#include "graphlab.hpp"
#include "GlobalHashMap.hpp"

DEFINE_bool( metrics, false, "Dump metrics");

DEFINE_int32(scale, 10, "Log2 number of vertices.");
DEFINE_int32(edgefactor, 16, "Average number of edges per vertex.");

DEFINE_int32(trials, 1, "Number of timed trials to run and average over.");

DEFINE_string(path, "", "Path to graph source file.");
DEFINE_string(format, "bintsv4", "Format of graph source file.");
DEFINE_string(seedsPath, "", "Path to the seeds");
DEFINE_string(outputPath, "./result", "Path and file name to the result.");

GRAPPA_DEFINE_METRIC(SimpleMetric<double>, init_time, 0);
GRAPPA_DEFINE_METRIC(SimpleMetric<double>, tuple_time, 0);
GRAPPA_DEFINE_METRIC(SimpleMetric<double>, construction_time, 0);
GRAPPA_DEFINE_METRIC(SummarizingMetric<double>, total_time, 0);

const double alpha = 0.15;
const double alpha_com = .85 ; // 1- alpha
const int Unknown = -1;
const int number_of_groups = 2;
DEFINE_double(tolerance, 1.0E-2, "tolerance");
#define TOLERANCE FLAGS_tolerance

// Graph label data
struct CCData : public GraphlabVertexData<CCData> {
  int label;
  VertexID id;
};

struct label_counter {
  double label_count[number_of_groups];
  int nadj;
  
  explicit label_counter(int label, int _nadj) {
    CHECK(_nadj >= 0) << "We expect all nodes to have at least one neighbour";
    CHECK(label < number_of_groups) << "Label should be less than supported number of groups.";
    for (int i =0 ; i < number_of_groups; i++) {
       label_count[i] = 0.0;
    }

    nadj = _nadj;
    if (label != Unknown)
      label_count[label] = 1.0;
  }
  
  explicit label_counter() {
    for (int i =0 ; i < number_of_groups; i++) {
       label_count[i] = 0.0;
    }
  }
  
  explicit label_counter(int _nadj) {
    CHECK(_nadj >= 0) << "We expect all nodes to have at least one neighbor";
    
    nadj = _nadj;
    // Init the array
    for (int i =0 ; i < number_of_groups; i++) {
       label_count[i] = 0.0;
    }
  }

  void addValue(int index, double value) {
    CHECK( index > -1 && index < number_of_groups);
    label_count[index] = value;
  }

  label_counter& operator+=(const label_counter& other) {
    //CHECK(other.nadj > 0) << "We need to havee at least one neighbor.";
    for (int i = 0; i < number_of_groups; i++) {
      label_count[i] += (other.label_count[i])/other.nadj;

       // If summation is greater than one just set it to 1
      if (label_count[i] > 1)
        label_count[i] = 1;
    }

    return *this; 
  }
};

using G = Graph<CCData,Empty>;

struct LabelPropagation : public GraphlabVertexProgram<G, label_counter> {
  bool do_scatter;
  label_counter changelabel;
  label_counter emptyLabel;

  LabelPropagation(Vertex& v) {
    do_scatter = false;
    changelabel = label_counter(v.nadj);
    emptyLabel = label_counter(v.nadj);
  }

  bool gather_edges(const Vertex& v) const { return true; }

  Gather gather(const Vertex& v, Edge& e) const { 
    return label_counter(v->label, v.nadj); 
  }

  void apply(Vertex& v, const Gather& total) {
    bool hasProbability = false;

    // Still we need to find the maximum probability so we can find the label
    for (int i = 0; i < number_of_groups; i++) {
        if (total.label_count[i] > 0) {
          hasProbability = true;
          break;
        }
    }

    // All of neighbours are unknown, or it doesn't have neighbour
    if (!hasProbability) {
      //LOG(INFO) << "vertex id:  unknown " << v->id << " Label is unknown.";
      do_scatter = false;
      v->activate();
    } else {
        for (int i = 0; i < number_of_groups; i++)
          changelabel.addValue(i, total.label_count[i]);

        do_scatter = true;
      }
  }

  bool scatter_edges(const Vertex& v) const { return do_scatter; }
  
  Gather scatter(const Edge& e, Vertex& target) const {
    // if target is still active do the scatter otherwise we scatter an 
    // empty label_counter, which has zero affect on the target.
    if (target->active)
      return changelabel;
    return emptyLabel;
  }
};

std::map<int64_t, int> loadSeed(std::string path) {
  // Load the seeds
  std::ifstream file(path);

  std::map<int64_t , int> seeds;
  while (file.good()) {
      std::string line;
      std::getline(file, line);
      std::string strLabel;
      std::string strId;
      int id;
      int label;

      std::stringstream lineStream(line);
      std::getline(lineStream, strId, ' ');
      std::getline(lineStream, strLabel, ' ');
      id = std::atoi(strId.c_str());
      CHECK(id >= 0);
      label = std::atoi(strLabel.c_str());
      //label -= 1;
     // LOG(INFO) << "Seed label is: " << label;
      seeds[id] = label;
  }
  LOG(INFO) << "Loaded Seed file. Count: " << seeds.size();

  return seeds;
}

int main(int argc, char* argv[]) {

  init(&argc, &argv);
  run([]{

    double t;

    TupleGraph tg;

    GRAPPA_TIME_REGION(tuple_time) {
      if (FLAGS_path.empty()) {
        LOG(INFO) << "We need to have a path to a graph.";
      } else {
        LOG(INFO) << "loading " << FLAGS_path;
        tg = TupleGraph::Load(FLAGS_path, FLAGS_format);
      }
    }

    LOG(INFO) << tuple_time;
    LOG(INFO) << "constructing graph";
    t = walltime();

    auto g = G::Undirected(tg);
    LOG(INFO) << "Finished loading graph";
    construction_time = walltime()-t;
    LOG(INFO) << construction_time;

    for (int i = 0; i < FLAGS_trials; i++) {
      if (FLAGS_trials > 1) LOG(INFO) << "trial " << i;
      // Load seed file
      LOG(INFO) << "start loading seed file";
      {
      symmetric_static std::map<int64_t, int> symmetric_seeds;
          on_all_cores( []{
              symmetric_seeds = loadSeed(FLAGS_seedsPath);
            });

        LOG(INFO) << "Init the labels";
        LOG(INFO) << "Size of the graph is: " << g->nv;
        forall(g, [](VertexID i, G::Vertex& v){
          if (symmetric_seeds.find(i) == symmetric_seeds.end())
          {
            // TODO: we don't need this anymore
            v->label = -1.0;
            v->activate(); // We only activate the unknown label, we don't want to change the seed nodes.
          } else {
            v->label = symmetric_seeds[i];
            v->deactivate(); // Keep the seeds deactivated, so we don't change them.
          }
        });
      }

      GRAPPA_TIME_REGION(total_time) {
       // LOG(INFO) << "Total unlabeled is:" << totalUnlabeled;
        LOG(INFO) << "Init is complete";
		NaiveGraphlabEngine<G,LabelPropagation>::OutputPath = FLAGS_outputPath;
        NaiveGraphlabEngine<G,LabelPropagation>::run_sync(g);
      }
    }

    LOG(INFO) << total_time;

    // for easy parallel writes, we'll make a separate file per core
    // {
      // symmetric_static std::ofstream myFile;
      // int pid = getpid();
      // on_all_cores( [pid] {
          // std::ostringstream oss;
          // oss << FLAGS_outputPath << "-" << pid << "-" << mycore();
          // new (&myFile) std::ofstream(oss.str());
          // if (!myFile.is_open()) exit(1);
        // });
      // forall(g, [](VertexID i, G::Vertex& v){ 
          // // LOG(INFO) << "id: " << i << " label: " << v->label;
          // myFile << i << " " << v->label << "\n";
        // });
      // on_all_cores( [] {
          // myFile.close();
        // });
    // }

    if (FLAGS_metrics) Metrics::merge_and_print();
    else { std::cerr << total_time << "\n" << iteration_time << "\n"; }
    Metrics::merge_and_dump_to_file();
  });
  finalize();
}
