#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <random>
#include <chrono>
#include <thread>
#include <algorithm>
//#include <memory>
//#include <atomic>

#include <math.h>
#include <stdint.h>

namespace semat {

// Semat : Sparse Efficient Model Allocation Topic

class Semat {
private:
    int K;
    int V;
    int M;

    float alpha;
    float beta;
    
    int iterations;
    int num_cores;

    std::vector<std::vector<int>> data;
    std::unordered_map<std::string, int> dict_map;
    std::vector<std::string> dict_vec;

    std::vector<std::vector<int>> Z;

    std::vector<std::unordered_map<int, int>> nv;
    std::vector<std::unordered_map<int, int>> nm;
    std::vector<int> nvsum;
    std::vector<int> nmsum;

    float s_;
    int cnt_;

    std::vector<std::mt19937> G;
    std::vector<std::uniform_real_distribution<float>> D;

public:
    Semat(int t = 128, float a = 0.1, float b = 0.01,
          int iters = 1000, int num_cores = 8) 
        : K(t), alpha(a), beta(b), iterations(iters), num_cores(num_cores) {
        
        auto s = std::chrono::system_clock::now().time_since_epoch().count();
        for (int i = 0; i < num_cores; i++) {
            G.emplace_back(s + i);
            D.emplace_back(0.0F, 1.0F);
        }
    } 

    bool LoadCorpus(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error open : " << filename << std::endl;
            return false;
        }

        std::string line;
        std::vector<std::vector<int>> temp_data;

        while (std::getline(file, line)) {
            if (line.empty()) continue;

            std::vector<int> document;
            std::istringstream iss(line);
            std::string token;

            while (iss >> token) {
                if (dict_map.find(token) == dict_map.end()) {
                    int index = dict_vec.size();
                    dict_map[token] = index;
                    dict_vec.push_back(token);
                }
                document.push_back(dict_map[token]);
            }

            if (!document.empty()) {
                temp_data.push_back(std::move(document));
            }
        }

        data = std::move(temp_data);
        file.close();

        V = dict_vec.size();
        M = data.size();

        std::cout << "Corpus Load: " << M << " documents, "
                  << V << " vocabulary size, using "
                  << num_cores << " Cpu cores." << std::endl;
        return true;
    }

    void Init() {
        nv.resize(V);
        nm.resize(M);
        nvsum.resize(K, 0);
        nmsum.resize(M, 0);
        Z.resize(M);

        std::uniform_int_distribution<int> T(0, K-1);

        for (int m = 0; m < M; m++) {
            int N = data[m].size();
            Z[m].resize(N);

            for (int i = 0; i < N; i++) {
                int w = data[m][i];
                int t = T(G[0]);
                Z[m][i] = t;

                nv[w][t]++;
                nm[m][t]++;
                nvsum[t]++;
                nmsum[m]++;
                cnt_ += 1;
            }
        }
    }

    void UpdateCache() {
        s_ = 0.0;
        for (int t = 0; t < K; t++) {
            s_ += alpha*beta/(nvsum[t] + beta*V);
        }
    }

    int SparseSample(int core, int m, int w) {
        float s = s_;

        float r = 0.0;
        for (const auto& kv : nm[m]) {
            int t = kv.first;
            int nt_m = kv.second;
            r += nt_m * beta /(nvsum[t] + beta*V);
        }

        float q = 0.0;
        for (const auto& kv : nv[w]) {
            int t = kv.first;
            int nw_t = kv.second;

            int nt_m = 0;
            auto it = nm[m].find(t);
            if (it != nm[m].end()) {
                nt_m = it->second;
            }

            q += (alpha + nt_m) * nw_t / (nvsum[t] + beta*V);
        }

        float a = s + r + q;
        float u = D[core](G[core])*a;

        if (u < s) {
            float e = u;
            float cum = 0.0;
            for (int t = 0; t < K; t++) {
                cum += alpha*beta / (nvsum[t] + beta*V);
                if (e <= cum) {
                    return t;
                }
            }
        } else if (u < s + r) {
            float e = u - s;
            float cum = 0.0;
            for (const auto& kv : nm[m]) {
                int t = kv.first;
                int nt_m = kv.second;
                cum += nt_m*beta/(nvsum[t] + beta*V);
                if (e <= cum) {
                    return t;
                }
            }
        } else {
            float e = u - s - r;
            float cum = 0.0;
            for (const auto& kv : nv[w]) {
                int t = kv.first;
                int nw_t = kv.second;

                int nt_m = 0;
                auto it = nm[m].find(t);
                if (it != nm[m].end()) {
                    nt_m = it->second;
                }

                cum += (alpha + nt_m) * nw_t / (nvsum[t] + beta*V);
                if (e <= cum) {
                    return t;
                }
            }
        }

        return static_cast<int>(D[core](G[core])*K);
    }

    void UpdateCount(int m, int w, int ot, int nt) {
        if (ot == nt) return;

        auto& doc_counts = nm[m];
        doc_counts[ot]--;
        if (doc_counts[ot] == 0) {
            doc_counts.erase(ot);
        }
        doc_counts[nt]++;

        auto& token_counts = nv[w];
        token_counts[ot]--;
        if (token_counts[ot] == 0) {
            token_counts.erase(ot);
        }
        token_counts[nt]++;

        nvsum[ot]--;
        nvsum[nt]++;
    }

    void RunSample() {
        auto start = std::chrono::high_resolution_clock::now();

        int doc_block_size = (M + num_cores - 1)/num_cores;
        int token_block_size = (V + num_cores - 1)/num_cores;

        for (int iter = 0; iter < iterations; iter++) {
            if (iter % 10 == 0) {
                std::cout << "Iteration " << iter+1 << "/" << iterations << std::endl;
            }

            UpdateCache();

            for (int pass = 0; pass < num_cores; pass++) {
                std::vector<std::thread> workers;

                for (int t = 0; t < num_cores; t++) {
                    workers.emplace_back([&, t, pass]() {
                        int doc_block_id = t;
                        int token_block_id = (t + pass) % num_cores;

                        int doc_start = doc_block_id * doc_block_size;
                        int doc_end = std::min(doc_start+doc_block_size, M);
                        int token_start = token_block_id * token_block_size;
                        int token_end = std::min(token_start+token_block_size, V);

                        for (int m = doc_start; m < doc_end; m++) {
                            for (int i = 0; i < data[m].size(); i++) {
                                int w = data[m][i];

                                if (w >= token_start && w < token_end) {
                                    int ot = Z[m][i];
                                    int nt = SparseSample(t, m, w);
                                    Z[m][i] = nt;
                                    UpdateCount(m, w, ot, nt);
                                }
                            }
                        }
                    });
                }

                for (auto& worker: workers) {
                    worker.join();
                }
            }

            if ((iter + 1)%10 == 0 || iter == iterations -1) {
                float r = LogLikelihood();
                float token_count = cnt_;
                if (token_count > 0) {
                    float p = std::exp(-r/token_count);

                    int active_count = 0;
                    for (int k = 0; k < K; k++) {
                        if (nvsum[k] > 0) active_count++;
                    }

                    std::cout << "Iteration " << iter + 1
                              << ": Perplexity = " << p 
                              << ", Active topics = " << active_count << "/" << K 
                              << std::endl;
                }
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> elapsed = end - start;
        std::cout << "N-Queens SparseLDA Completed in " 
                  << elapsed.count() << " seconds " << std::endl;
    }

    float LogLikelihood() {
        float r = 0.0;
        for (int m = 0; m < M; m++) {
            for (int i = 0; i < data[m].size(); i++) {
                int w = data[m][i];
                int t = Z[m][i];
                // P(t/m) = (m_mt + a) / (n_m + Ka)
                int n_mt = 0;
                auto it_nm = nm[m].find(t);
                if (it_nm != nm[m].end()) {
                    n_mt = it_nm->second;
                }
                float p_t_given_m = (n_mt + alpha) /(nmsum[m] + K*alpha);

                // P(w|t) = (n_wt + b) / (n_t + Vb)
                int n_wt = 0;
                auto it_nv = nv[w].find(t);
                if (it_nv != nv[w].end()) {
                    n_wt = it_nv->second;
                }
                float p_w_given_t = (n_wt + beta) / (nvsum[t] + V*beta);

                // log P(w|m) = log P(t|m) + log(w|t)
                r += std::log(p_t_given_m) + std::log(p_w_given_t);
            }
        }
        return r;
    }

    void SaveModel(const std::string& name) {
        std::ofstream vocab_file(name + ".vocab");
        for (const auto& word : dict_vec) {
            vocab_file << word << std::endl;
        }
        vocab_file.close();
        
        std::ofstream phi_file(name + ".phi");
        for (int k = 0; k < K; k++) {
            if (nvsum[k] == 0) continue; 
            
            phi_file << "Topic " << k << ":" << std::endl;
            
            std::vector<std::pair<float, std::string>> word_probs;
            for (int w = 0; w < V; w++) {
                int count = 0;
                auto it = nv[w].find(k);
                if (it != nv[w].end()) {
                    count = it->second;
                }
                if (count > 0) { 
                    float prob = (count + beta) / (nvsum[k] + V * beta);
                    word_probs.emplace_back(prob, dict_vec[w]);
                }
            }
            
            std::sort(word_probs.begin(), word_probs.end(), 
                      [](const std::pair<float, std::string>& a, const std::pair<float, std::string>& b) {
                          return a.first > b.first;
                      });
            
            for (const auto& wp : word_probs) {
                phi_file << wp.second << ":" << wp.first << " ";
            }
            phi_file << std::endl << std::endl;
        }
        phi_file.close();
        
        std::ofstream theta_file(name + ".theta");
        for (int m = 0; m < M; m++) {
            theta_file << "Doc " << m << ": ";
            
            std::vector<std::pair<int, float>> topic_probs;
            for (const auto& kv : nm[m]) {
                int k = kv.first;
                int count = kv.second;
                float prob = (count + alpha) / (nmsum[m] + K * alpha);
                topic_probs.emplace_back(k, prob);
            }
            
            std::sort(topic_probs.begin(), topic_probs.end());
            
            for (const auto& tp : topic_probs) {
                theta_file << "topic" << tp.first << ":" << tp.second << " ";
            }
            theta_file << std::endl;
        }
        theta_file.close();
        
        std::cout << "Model saved to " << name << ".* files" << std::endl;
    }
};




} // namespace semat

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << "data_file [topics=128] [iters=10] [a=0.1] [b=0.01] [num_cores=8]";
        return 1;
    }

    std::string filename = argv[1];
    int topics = (argc > 2) ? std::stoi(argv[2]) : 128;
    int iters = (argc > 3) ? std::stoi(argv[3]) : 10;
    float alpha = (argc > 4) ? std::stof(argv[4]) : 0.1F;
    float beta = (argc > 5) ? std::stof(argv[5]) : 0.01F;
    int num_cores = (argc > 6) ? std::stoi(argv[6]) : std::thread::hardware_concurrency()/2;

    semat::Semat se(topics, alpha, beta, iters, num_cores); 

    if (!se.LoadCorpus(filename)) {
        return 1;
    }

    se.Init();
    se.RunSample();
    se.SaveModel("semat"); 

    return 0;
}