#include <iostream>
#include "cppflow/cppflow.h"


void readLabels(const char *labelsFile, std::vector<std::string> &labels)
{
    std::string line;
    std::ifstream fin(labelsFile);
    while (getline(fin, line)) {
        labels.push_back(line);
    }
}

int main(int argc,char ** argv) {

    if (argc != 4)
    {
        fprintf(stderr, "Parameters: modelfile labels image\n");
        exit(-1);
    }


    chrono::steady_clock::time_point Tbegin, Tend;
	
	const int NUM_CLASSES = 53;
    const char *MODEL_FILE = argv[1];
    const char *LABELS_FILE = argv[2];
    const char *IMG_FILE = argv[3];

    std::vector<std::string> labels;
    readLabels(LABELS_FILE, labels);

    auto input = cppflow::decode_jpeg(cppflow::read_file(std::string(IMG_FILE)));

    input = cppflow::cast(input, TF_UINT8, TF_FLOAT);
    input = cppflow::expand_dims(input, 0);
    
    Tbegin = chrono::steady_clock::now();
    cppflow::model model(MODEL_FILE);
    auto output = model(input);
    Tend = chrono::steady_clock::now();

    auto idx = cppflow::arg_max(output, 1);
    std::string label = labels[idx];
    
    std::cout << IMG_FILE << ":" << label << " " << cppflow::arg_max(output, 1) <<  cppflow::max(output,1) << std::endl;

    //calculate time
    f = chrono::duration_cast <chrono::milliseconds> (Tend - Tbegin).count();
    cout << "Process time: " << f << " ms" << endl;
 
    return 0;
}