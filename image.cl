__kernel void sum(__global float *A, __global float *output, ulong AOffset, __local float *target) {
    const size_t globalId = get_global_id(0);
    const size_t localId = get_local_id(0);
    target[localId] = A[globalId + AOffset];

    barrier(CLK_LOCAL_MEM_FENCE);
    size_t blockSize = get_local_size(0);
    size_t halfBlockSize = blockSize / 2;
    while (halfBlockSize > 0) {
        if (localId < halfBlockSize) {
            target[localId] += target[localId + halfBlockSize];
            if ((halfBlockSize * 2) < blockSize) { // uneven block division
                if (localId == 0) {                // when localID==0
                    target[localId] += target[localId + (blockSize - 1)];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        blockSize = halfBlockSize;
        halfBlockSize = blockSize / 2;
    }
    if (localId == 0) {
        output[get_group_id(0)] = target[0];
    }
}

// https://sodocumentation.net/opencl/topic/5893/pseudo-random-number-generator-kernel-example
uint wang_hash(uint seed) {
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return seed;
}

void wang_rnd_0(__global unsigned int *rnd_buffer, int id) {
    uint maxint = 0;
    maxint--;
    uint rndint = wang_hash(id);
    rnd_buffer[id] = rndint;
}

float wang_rnd(__global unsigned int *rnd_buffer, int id) {
    uint maxint = 0;
    maxint--; // not ok but works
    uint rndint = wang_hash(rnd_buffer[id]);
    rnd_buffer[id] = rndint;
    return ((float)rndint) / (float)maxint;
}

__kernel void rnd_init(__global unsigned int *rnd_buffer) {
    int id = get_global_id(0);
    wang_rnd_0(rnd_buffer, id); // each (id) thread has its own random seed now
}

__kernel void rnd_1(__global unsigned int *rnd_buffer) {
    int id = get_global_id(0);

    // can use this to populate a buffer with random numbers
    // concurrently on all cores of a gpu
    float thread_private_random_number = wang_rnd(rnd_buffer, id);
}

// https://stackoverflow.com/questions/9912143/how-to-get-a-random-number-in-opencl
int rand(int* seed) // 1 <= *seed < m
{
    int const a = 16807; //ie 7**5
    int const m = 2147483647; //ie 2**31-1

    *seed = (long(*seed * a))%m;
    return(*seed);
}

kernel random_number_kernel(global int* seed_memory)
{
    int global_id = get_global_id(1) * get_global_size(0) + get_global_id(0); // Get the global id in 1D.

    // Since the Park-Miller PRNG generates a SEQUENCE of random numbers
    // we have to keep track of the previous random number, because the next
    // random number will be generated using the previous one.
    int seed = seed_memory[global_id];

    int random_number = rand(&seed); // Generate the next random number in the sequence.

    seed_memory[global_id] = *seed; // Save the seed for the next time this kernel gets enqueued.
}