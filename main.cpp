#include <iostream>
#include <fstream>
#include <algorithm>
#include "cppflow/cppflow.h"

using namespace std;

int main(int argc,char ** argv) {

    if (argc != 4)
    {
        fprintf(stderr, "Parameters: modelfile labels image\n");
        exit(-1);
    }


//    chrono::steady_clock::time_point Tbegin, Tend;
	
    int f;
    const int NUM_CLASSES = 53;
    const char *MODEL_FILE = argv[1];
    const char *LABELS_FILE = argv[2];
    const char *IMG_FILE = argv[3];
    std::vector<float> predictions;

    string labels[53] = {"10c", "10d", "10h", "10s", 
                        "2c", "2d","2h", "2s",
                        "3c", "3d","3h", "3s",
                        "4c", "4d","4h", "4s",
                        "5c", "5d","5h", "5s",
                        "6c", "6d","6h", "6s",
                        "7c", "7d","7h", "7s",
                        "8c", "8d","8h", "8s",
                        "9c", "9d","9h", "9s",
                        "Ac", "Ad","Ah", "As",
                        "Jc", "Jd","Jh", "Js",
                        "Kc", "Kd","Kh", "Ks",
                        "Qc", "Qd","Qh", "Qs","Zx"};


    auto input = cppflow::decode_jpeg(cppflow::read_file(std::string(IMG_FILE)));

    input = cppflow::cast(input, TF_UINT8, TF_FLOAT);
    input = cppflow::expand_dims(input, 0);
    
    //Tbegin = chrono::steady_clock::now();
    cppflow::model model(MODEL_FILE);
    auto outputData = model(input);
    //Tend = chrono::steady_clock::now();

    predictions = outputData.get_data<float>();
    auto proba = std::max_element(predictions.begin(), predictions.end());
    int idx = std::distance(predictions.begin(), proba);
    
    std::cout<<"Position of the label: " << idx <<std::endl;

    std::cout << IMG_FILE << ":" << " " << labels[idx] << " " <<  *proba*100 << "% " << std::endl;

    //std::cout << IMG_FILE << ":" << " " << cppflow::arg_max(outputData, 1) <<  cppflow::max(outputData,1) << std::endl;
    
    //calculate time
    //f = chrono::duration_cast <chrono::milliseconds> (Tend - Tbegin).count();
    //cout << "Process time: " << f << " ms" << endl;
 
    return 0;
}