#include <mpi.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include <set>
#include <queue>

using namespace std;

struct HeapItem
{
    vector<unsigned>::iterator cur, last;
    void next() { cur++; }
    unsigned get() const { return *cur; }
    bool has() const { return cur != last; }
    friend bool operator<(const HeapItem& l, const HeapItem& r) { return l.get() < r.get(); }
};

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    unsigned n, n_adjusted;
    vector<unsigned> v, local_v;

    if (world_rank == 0) {
        cin >> n;
        n_adjusted = (n + world_size - 1) / world_size * world_size;
        v.resize(n_adjusted);
        unsigned mul = 1664525, add = 1013904223, cur = 123456789;
        for (unsigned i = 0; i < n; i++) {
            v[i] = cur = cur * mul + add;
        }
        for (unsigned i = n; i < n_adjusted; i++) {
            v[i] = numeric_limits<unsigned>::max();
        }
        cerr << clock() * 1.0 / CLOCKS_PER_SEC << " s - finished generation\n";
    }

    MPI_Bcast(&n_adjusted, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    unsigned local_n = n_adjusted / world_size;
    local_v.resize(local_n);

    MPI_Scatter(v.data(), local_n, MPI_UNSIGNED, local_v.data(), local_n, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    sort(local_v.begin(), local_v.end());

    MPI_Gather(local_v.data(), local_n, MPI_UNSIGNED, v.data(), local_n, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        cerr << clock() * 1.0 / CLOCKS_PER_SEC << " s - finished sort of parts\n";
        vector<unsigned> res(n);
        vector<HeapItem> heap;
        for (unsigned i = 0; i < world_size; i++)
        {
            HeapItem item{ v.begin() + (unsigned long long)n_adjusted * i / world_size, v.begin() + (unsigned long long)n_adjusted * (i + 1) / world_size };
            if (item.has())
                heap.push_back(item);
        }
        make_heap(heap.begin(), heap.end());
        for (unsigned i = 0; i < n; i++)
        {
            pop_heap(heap.begin(), heap.end());
            HeapItem& item = heap.back();
            res[i] = item.get();
            item.next();
            if (item.has())
                push_heap(heap.begin(), heap.end());
            else
                heap.pop_back();
        }
        cerr << clock() * 1.0 / CLOCKS_PER_SEC << " s - finished merge\n";
        for (unsigned i = 0; i <= 99; i++)
            cout << res[(unsigned long long)i * n / 99] << " ";
        cerr << clock() * 1.0 / CLOCKS_PER_SEC << " s - finished output\n";
    }

    MPI_Finalize();
    return 0;
}
