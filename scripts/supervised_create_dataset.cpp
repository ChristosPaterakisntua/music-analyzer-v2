// run supervised_create_dataset.exe to create dataset
// run supervised_create_dataset.exe --del-parsed to delete parsed mp3s

#include <iostream>
#include <string>
#include <filesystem>
#include <cstdlib> // system()


const std::string & REPO_ROOT = std::filesystem::current_path().parent_path().generic_string();
const std::string & SCRIPTS_DIR = REPO_ROOT + "/scripts";
const std::string & PARSED_DATASET_DIR = REPO_ROOT + "/parsed_dataset";
const std::string & SRC_DIR = REPO_ROOT + "/src/music_analyzer_v2";
const std::string & RUN_CMD = 
    "cd /d " + SRC_DIR + " && python create_dataset.py";
std::string CLEAR_CMD = "cd /d " + SCRIPTS_DIR + " && python delete_parsed_mp3.py";
const int MAX_NO_PROGRESS = 3;


size_t count_npz() {
    size_t n = 0;
    for (auto& e : std::filesystem::directory_iterator(PARSED_DATASET_DIR)) {
        if (e.is_regular_file() && e.path().extension() == ".npz") {
            ++n;
        }
    }
    return n;
}


int main(int argc, char* argv[]) {
    
    // deletes parsed mp3
    if (argv[1] == "--del-parsed") {
        CLEAR_CMD += " --confirm";
    }

    int no_progress = 0; // no progress crashes counter
    while (true) {
        size_t before = count_npz();
        std::cout << "[watch] .npz now: " << before << " -> running...\n";
        system(CLEAR_CMD.c_str());
        int rc = system(RUN_CMD.c_str()); // blocks until python is finished
        size_t after = count_npz();

        // clean exit with no new work
        if (rc == 0 && after == before) {
            std::cout << "[watch] DONE: clean exit, no new work!\n";
            system(CLEAR_CMD.c_str());
            return 0;
        }

        // progress was made rerun and zero no_progress counter
        if (after > before) {
            std::cout << "[watch] progress " << before << " -> " << after
                      << ", restarting...\n";
            no_progress = 0;
            continue;
        }

        // crash eith no progress. If more than max alert & exit
        if (++no_progress > MAX_NO_PROGRESS) {
            std::cerr << "[watch] STUCK: no progress after "
                      << MAX_NO_PROGRESS << " restarts.\n";
            return 1;
        }

        // error in python
        if (rc) {
            std::cout << "[watch] crash without progeress (exit " << rc
                      << "), restart " << no_progress << "/" << MAX_NO_PROGRESS
                      << '\n';
        }

    }
}