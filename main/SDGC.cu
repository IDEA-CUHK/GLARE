#pragma once


#include <CLI11/CLI11.hpp>
#include <SNIG/SNIG.hpp>
#include <SNIG_GLARE/SNIG_GLARE.hpp>
#include <XY/XY.hpp>
#include <XY_GLARE/XY_GLARE.hpp>
#include <SNICIT/SNICIT.hpp>
#include <SNICIT_GLARE/SNICIT_GLARE.hpp>
#include <BF_GLARE/BF_GLARE.hpp>
#include <BF/BF.hpp>
#include <iostream>

int main(int argc, char* argv[]) {

  CLI::App app{"SDGC23"};

  std::string mode = "BFAug";
  app.add_option(
    "-m, --mode", 
    mode, 
    "select mode(SNICIT, BF, SNIG, XY, or SNICIT_Aug), default is SNICIT"
  );


  size_t num_neurons = 1024;
  app.add_option(
    "-n, --num_neurons", 
    num_neurons, 
    "total number of neurons, default is 1024"
  );

  size_t num_layers = 120;
  app.add_option(
    "-l, --num_layers",
    num_layers, 
    "total number of layers, default is 120"
  );

  int threshold = 30;
  app.add_option(
    "-t, --threshold",
    threshold,
    "number of threshold, default is 30"
  );
  size_t input_batch_size = 60000;
  app.add_option(
    "-b, --input_batch_size", 
    input_batch_size,
    "number of input bath size, default is 60000, must be a factor of num_input (60000)"
  );
  std::map<int, float> bias_map = {
      {65536, -0.45},
      {16384, -0.4},
      {4096, -0.35},
      {1024, -0.3}
  };

  CLI11_PARSE(app, argc, argv);

  std::string weight_path("../dataset/SDGC/tsv_weights/neuron"+std::to_string(num_neurons));

  std::string input_path("../dataset/SDGC/MNIST/sparse-images-"+std::to_string(num_neurons)+".tsv");

  std::string golden_path("../dataset/SDGC/MNIST/neuron"+std::to_string(num_neurons)+"-l"+std::to_string(num_layers)+"-categories.tsv");
  
  float bias = bias_map[num_neurons];


  std::cout << "Current mode: " << mode << std::endl;

  if(mode == "BF") {
    GLARE::BF bf(
      weight_path, 
      bias,
      num_neurons, 
      num_layers
    );
    bf.infer(input_path, golden_path, 60000);
  }
  else if(mode == "BF_GLARE") {
    GLARE::BF_GLARE bfglare(
      weight_path, 
      bias,
      num_neurons, 
      num_layers
    );
    bfglare.infer(input_path, golden_path, 60000);
  }
  else if(mode == "SNIG") {
    GLARE::SNIG snig(
      weight_path, 
      bias,
      num_neurons, 
      num_layers
    );
    snig.infer(input_path, golden_path, 60000, 60000, 2);
  }
  else if(mode == "SNIG_GLARE") {
    GLARE::SNIG_GLARE snigglare(
      weight_path, 
      bias,
      num_neurons, 
      num_layers
    );
    snigglare.infer(input_path, golden_path, 60000, 60000, 2);
  }
  else if(mode == "XY") {
    GLARE::XY xy(
      weight_path, 
      bias,
      num_neurons, 
      num_layers,
      threshold
    );
    xy.infer(input_path, golden_path, 60000, input_batch_size);
  }
  else if(mode == "XY_GLARE") {
    GLARE::XY_GLARE xyglare(
      weight_path, 
      bias,
      num_neurons, 
      num_layers,
      threshold
    );
    xyglare.infer(input_path, golden_path, 60000, input_batch_size);
  }
  else if(mode == "SNICIT") {
    GLARE::SNICIT snicit(
      weight_path, 
      bias,
      num_neurons, 
      num_layers,
      threshold
    );
    snicit.infer(input_path, golden_path, 60000, input_batch_size);
  }
  else if(mode == "SNICIT_GLARE") {
    GLARE::SNICIT_GLARE snicitglare(
      weight_path, 
      bias,
      num_neurons, 
      num_layers,
      threshold
    );
    snicitglare.infer(input_path, golden_path, 60000, input_batch_size);
  }
  else {
    using namespace std::literals::string_literals;
    throw std::runtime_error("Error mode. Please correct your mode name"s);
  }


  return 0;
}
