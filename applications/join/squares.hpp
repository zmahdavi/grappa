#pragma once
#include "Query.hpp"
#include "../graph500/grappa/graph.hpp"

class SquareQuery : public Query {
  public:
    virtual void preprocessing(std::vector<tuple_graph> relations);

    virtual void execute(std::vector<tuple_graph> relations);
};
