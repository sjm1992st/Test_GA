#include <iostream>
#include <ctime>
#include <algorithm>
#include <functional>
#include <iomanip>
#include <fstream>  
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math_constants.h>
#include <sstream>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

#include "constants.h"
using namespace std;
// assume block size equal population size
//模板函数：将string类型变量转换为常用的数值类型 
template <class Type>
Type stringToNum(const string& str){
	istringstream iss(str);
	Type num;
	iss >> num;
	return num;
}

template <class T>
int getArrayLen(T& array)
{
	return (sizeof(array) / sizeof(array[0]) - 1);
}

const int THREADS_PER_BLOCK = 256*2;

void cudasafe(cudaError_t error, char* message = "Error occured") {
	if(error != cudaSuccess) {
		fprintf(stderr,"ERROR: %s : %i\n", message, error);
		exit(-1);
	}
}

__global__ void randomInit(curandState* state, unsigned long seed) {
    int tid = threadIdx.x;
    curand_init(seed, tid, 0, state + tid);
}

__device__ float fitness(M_args deviceParameter, M_args_Tset deviceParameter_Tset,float tau) {
    float result = 0;
	//printf("%d_a",deviceParameter.spike_data_num);
	//printf("%d_b", deviceParameter_Tset.length);

	for (size_t i = 0; i<deviceParameter.spike_data_num; ++i)
		for (size_t j = 0; j<deviceParameter_Tset.length; ++j)
	{
			//printf("%f_c ", deviceParameter.spike_data[i]);
			//printf("%f_d ", deviceParameter_Tset.spike_TestData[j]);
			result += expf(-fabsf(deviceParameter.spike_data[i] - deviceParameter_Tset.spike_TestData[j])*1.0/tau);
			//printf("%f_1 ", result);
       // ++curPos;
    }
	for (size_t i = 0; i<deviceParameter.spike_data_num; ++i)
		for (size_t j = 0; j<deviceParameter.spike_data_num; ++j)
		{
			result += expf(-fabsf(deviceParameter.spike_data[i] - deviceParameter.spike_data[j])*1.0 / tau);
			//printf("%f_2 ", result);
			// ++curPos;
		}
	for (size_t i = 0; i<deviceParameter_Tset.length; ++i)
		for (size_t j = 0; j<deviceParameter_Tset.length; ++j)
		{
			result -= 2*expf(-fabsf(deviceParameter_Tset.spike_TestData[i] - deviceParameter_Tset.spike_TestData[j])*1.0 / tau);
			//printf("%f_3 ", result);
			// ++curPos;
		}
	//printf("%f_4 ", result);
    return result;
}

__device__ float rastrigin(const float *curPos) {
    float result = 10.0f * VAR_NUMBER;
    for (size_t i=0; i<VAR_NUMBER; ++i) {
        result += *curPos * *curPos - 10.0f * cosf(2 * CUDART_PI_F * *curPos);
        ++curPos;
    }
    return result;
}

////__global__ void GAKernel(float* population, float** sharedPopulation, float* sharedScore, M_args deviceParameter, M_args_Tset *deviceParameter_Tset, float tau) {
////	//__shared__ float sharedPopulation[THREADS_PER_BLOCK * 2][VAR_NUMBER];
////	//__shared__ float sharedScore[THREADS_PER_BLOCK * 2];
////
////
////	const int gid = blockDim.x * blockIdx.x + threadIdx.x;
////	const int tid = threadIdx.x;
////
////	M_args_Tset curPos = deviceParameter_Tset[tid];
////	sharedScore[tid] = fitness(deviceParameter, curPos, tau);
////	// loading initial random population into shared memory
////	if (gid < POPULATION_SIZE) {
////		for (int i = 0; i < VAR_NUMBER; ++i)
////			sharedPopulation[tid][i] = population[gid * VAR_NUMBER + i];
////	}
////
////	sharedScore[tid + THREADS_PER_BLOCK] = 123123.0;
////
////	__syncthreads();
////	//return 
////}
__global__ void GAKernel_GenEach(float* population, ScoreWithId* score, curandState* randomStates, M_args deviceParameter, M_args_Tset *deviceParameter_Tset, float tau, int genindex, int MaxGeneration) {
	// we first have to calculate the score for the first half of threads
	//const float *curPos = sharedPopulation[tid];
	__shared__ float sharedPopulation[THREADS_PER_BLOCK * 2][VAR_NUMBER];
	__shared__ float sharedScore[THREADS_PER_BLOCK * 2];
	const float SIGN[2] = { -1.0f, 1.0f };
	const float MULT[2] = { 1.0f, 0.0f };

	const int gid = blockDim.x * blockIdx.x + threadIdx.x;
	const int tid = threadIdx.x;

	// loading initial random population into shared memory
	if (gid < POPULATION_SIZE) {
		for (int i = 0; i<VAR_NUMBER; ++i)
			sharedPopulation[tid][i] = population[gid * VAR_NUMBER + i];
	}

	//sharedScore[tid + THREADS_PER_BLOCK] = 123123.0;

	__syncthreads();
	curandState &localState = randomStates[tid];
	M_args_Tset curPos = deviceParameter_Tset[tid];
	sharedScore[tid] = fitness(deviceParameter, curPos, tau);
	// calculating score for the second half of individuals
	M_args_Tset curPos_b = deviceParameter_Tset[tid + THREADS_PER_BLOCK];
	sharedScore[tid + THREADS_PER_BLOCK] = fitness(deviceParameter, curPos_b, tau);

	__syncthreads();

	// selection
	// first half of threads writes best individual into its position
	if (sharedScore[tid] > sharedScore[tid + THREADS_PER_BLOCK]) {
		for (int i = 0; i < VAR_NUMBER; ++i)
			sharedPopulation[tid][i] = sharedPopulation[tid + THREADS_PER_BLOCK][i];
		sharedScore[tid] = sharedScore[tid + THREADS_PER_BLOCK];
	}

	__syncthreads();

	// now we've got best individuals in the first half of sharedPopulation

	// crossovers
	const int first = curand_uniform(&localState) * THREADS_PER_BLOCK;
	const int second = curand_uniform(&localState) * THREADS_PER_BLOCK;

	const float weight = curand_uniform(&localState);
	for (int i = 0; i < VAR_NUMBER; ++i) {
		sharedPopulation[tid + THREADS_PER_BLOCK][i] = sharedPopulation[first][i] * weight + sharedPopulation[second][i] * (1.0f - weight);
	}

	__syncthreads();

	// mutations on second half of population
	if (curand_uniform(&localState) < 0.8) {
		const float order = (curand_uniform(&localState) * 17) - 15;
		for (int i = 0; i < VAR_NUMBER; ++i) {
			const float mult = MULT[curand_uniform(&localState) < 0.8f];
			const float sign = SIGN[curand_uniform(&localState) < 0.5f];
			const float order_deviation = (curand_uniform(&localState) - 0.5f) * 5;
			sharedPopulation[tid + THREADS_PER_BLOCK][i] += powf(10.0f, order + order_deviation) * sign * mult;
		}
	}

	//sharing a part of population with others
	if ((blockIdx.x + first) % 5 == 0) {
		for (int i = 0; i < VAR_NUMBER; ++i)
			population[gid * VAR_NUMBER + i] = sharedPopulation[tid][i];
	}

	// take some best individuals from neighbour
	if ((blockIdx.x + first) % 3 == 0) {
		if (curand_uniform(&localState) < 0.11) {
			const int anotherBlock = curand_uniform(&localState) * (POPULATION_SIZE / THREADS_PER_BLOCK);
			const int ngid = blockDim.x * anotherBlock + threadIdx.x;
			for (int i = 0; i < VAR_NUMBER; ++i)
				sharedPopulation[tid][i] = population[ngid * VAR_NUMBER + i];
			//sharedScore[tid] = fitness(sharedPopulation[tid], deviceParameter);
			//sharedScore[tid]=fitness(deviceParameter, curPos_b, tau);
		}
	}

	////// output current population back
	if (gid < POPULATION_SIZE) {
		for (int i = 0; i < VAR_NUMBER; ++i)
			population[gid * VAR_NUMBER + i] = sharedPopulation[tid][i];
		if (genindex>=MaxGeneration)
			score[gid].score = sharedScore[tid];
	}
}

void printPopulation(const float* devicePopulation, const ScoreWithId* deviceScore) {
	float population[POPULATION_SIZE][VAR_NUMBER];
	cudasafe(cudaMemcpy(population, devicePopulation, POPULATION_SIZE * VAR_NUMBER * sizeof(float), cudaMemcpyDeviceToHost), "Could not copy population from device");

	ScoreWithId score[POPULATION_SIZE];
	cudasafe(cudaMemcpy(score, deviceScore, POPULATION_SIZE * sizeof (ScoreWithId), cudaMemcpyDeviceToHost), "Could not copy score to host");

	//std::cout.cetf(std::ios::fixed);
	std::cout.precision(12);
	
	for (int i=0; i<POPULATION_SIZE; ++i) {
		std::cout << std::setw(15) << i << ' ';
	}
	std::cout << std::endl;

	for (int i=0; i<VAR_NUMBER; i++) {
		for (int u=0; u<POPULATION_SIZE; ++u) {
			std::cout << std::setw(15) << population[u][i] << ' ';
		}
		std::cout << std::endl;
	}
	std::cout << "Score: " << std::endl;
	for (int i=0; i<POPULATION_SIZE; ++i) {
		std::cout << std::setw(15) << score[i].score << ' ';
	}
	std::cout << std::endl;
}
void printFinalPopulation(const float* devicePopulation, const ScoreWithId* deviceScore) {
	float population[POPULATION_SIZE][VAR_NUMBER];
	cudasafe(cudaMemcpy(population, devicePopulation, POPULATION_SIZE * VAR_NUMBER * sizeof(float), cudaMemcpyDeviceToHost), "Could not copy population from device");

	ScoreWithId score[POPULATION_SIZE];
	cudasafe(cudaMemcpy(score, deviceScore, POPULATION_SIZE * sizeof (ScoreWithId), cudaMemcpyDeviceToHost), "Could not copy score to host");

	//std::cout.cetf(std::ios::fixed);
	std::cout.precision(12);

	for (int i = 0; i<1; ++i) {
		std::cout << std::setw(15) << i << ' ';
	}
	std::cout << std::endl;

	for (int i = 0; i<VAR_NUMBER; i++) {
		for (int u = 0; u<1; ++u) {
			std::cout << std::setw(15) << population[u][i] << ' ';
		}
		std::cout << std::endl;
	}
	std::cout << "Score: " << std::endl;
	for (int i = 0; i<1; ++i) {
		std::cout << std::setw(15) << score[i].score << ' ';
	}
	std::cout << std::endl;
}
double solveGPU(M_args Parameter_) {
    cudasafe(cudaSetDevice(0), "Could not set device 0");

	double ans = 0;
	int MaxGeneration = 1;
	float tau = 12;
	//M_args *IndexParameter_ = new M_args[MaxGeneration];
	//IndexParameter_ = 0;
	//IndexParameter_[0] = Parameter_;
	///////////////////////////////



	//////////////////////////////////
	float *population = new float[POPULATION_SIZE * VAR_NUMBER];

	for (int i=0; i<POPULATION_SIZE; ++i) {
		for (int j=0; j<VAR_NUMBER; ++j) {
			population[i * VAR_NUMBER + j] = (float_random() - 0.5f) * 10;
		}
	}
	M_args_Tset *Parameter_Tset=new M_args_Tset[POPULATION_SIZE];

	// copying population to device
	float *devicePopulation = 0;
	float *nextGeneration = 0;
	M_args_Tset *deviceParameter_Tset = 0;
	ScoreWithId *deviceScore = 0;
	curandState* randomStates;
	M_args deviceParameter_;
	deviceParameter_.current_data_num = Parameter_.current_data_num;
	deviceParameter_.spike_data_num = Parameter_.spike_data_num;
	//int DataLength = getArrayLen(Parameter_.spike_data);


	//int DataLengthC = getArrayLen(Parameter_.current_data);
	//Parameter_.length = DataLength;

	cudasafe(cudaMalloc(&randomStates, THREADS_PER_BLOCK * sizeof(curandState)), "Could not allocate memory for randomStates");
	cudasafe(cudaMalloc((void **)&devicePopulation, POPULATION_SIZE * VAR_NUMBER * sizeof(float)), "Could not allocate memory for devicePopulation");
	cudasafe(cudaMalloc((void **)&nextGeneration, POPULATION_SIZE * VAR_NUMBER * sizeof(float)), "Could not allocate memory for nextGeneration");
	cudasafe(cudaMalloc((void **)&deviceScore, POPULATION_SIZE * sizeof (ScoreWithId)), "Could not allocate memory for deviceScore");
	cudasafe(cudaMalloc((void **)&deviceParameter_Tset, 2*POPULATION_SIZE * sizeof (M_args_Tset)), "Could not allocate memory for deviceParameter_Tset");
	cudasafe(cudaMalloc((void **)&deviceParameter_.current_data, Parameter_.current_data_num*sizeof(float)), "Could not allocate memory for deviceParameter_");
	cudasafe(cudaMalloc((void **)&deviceParameter_.spike_data, Parameter_.spike_data_num*sizeof(float)), "Could not allocate memory for deviceParameter_");
	//cudasafe(cudaMalloc((void **)&deviceParameter_, sizeof(M_args)), "Could not allocate memory for deviceParameter_");
	//cudasafe(cudaMalloc((void **)&deviceParameter_.spike_TestData, DataLength*sizeof(float)), "Could not allocate memory for deviceParameter_");

	cudasafe(cudaMemcpy(devicePopulation, population, POPULATION_SIZE * VAR_NUMBER * sizeof(float), cudaMemcpyHostToDevice), "Could not copy population to device");
	cudasafe(cudaMemcpy(deviceParameter_.current_data, Parameter_.current_data, Parameter_.current_data_num*sizeof(float), cudaMemcpyHostToDevice), "Could not copy Parameter_current_data to device");
	cudasafe(cudaMemcpy(deviceParameter_.spike_data, Parameter_.spike_data, Parameter_.spike_data_num*sizeof(float), cudaMemcpyHostToDevice), "Could not copy Parameter_spike_data to device");

	//cudasafe(cudaMemcpy(deviceParameter_.spike_TestData, Parameter_.spike_TestData, DataLength*sizeof(float), cudaMemcpyHostToDevice), "Could not copy Parameter_ to device");

	// invoking random init
	randomInit<<<1, THREADS_PER_BLOCK>>>(randomStates, 900);
	cudasafe(cudaGetLastError(), "Could not invoke kernel randomInit");
	cudasafe(cudaDeviceSynchronize(), "Failed to syncrhonize device after calling randomInit");

	const int BLOCKS_NUMBER = (POPULATION_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;


	//__shared__ float sharedPopulation[THREADS_PER_BLOCK * 2][VAR_NUMBER];
	//__shared__ float sharedScore[THREADS_PER_BLOCK * 2];
    //for (int i=0; i<1115; i++) {
	//void GAKernel_GenEach(float* population, ScoreWithId* score, curandState* randomStates, M_args deviceParameter, M_args_Tset *deviceParameter_Tset, float tau)
	for (int k = 0; k < MaxGeneration; k++) 
	{
		for (int j = 0; j < POPULATION_SIZE; ++j)
		{
			float *temp_spike_TestData;
			
			Parameter_Tset[j].spike_TestData = HH_return(&population[j], VAR_NUMBER, Parameter_Tset[j].length);
			cudaMalloc(&temp_spike_TestData, Parameter_Tset[j].length*sizeof(float));
			std::cout << Parameter_Tset[j].length << std::endl;
			cudasafe(cudaMemcpy(&deviceParameter_Tset[j], &Parameter_Tset[j], sizeof (M_args_Tset), cudaMemcpyHostToDevice), "Could not copy deviceParameter_Tset1 to device");
			cudasafe(cudaMemcpy(temp_spike_TestData, Parameter_Tset[j].spike_TestData, (Parameter_Tset[j].length*sizeof(float)), cudaMemcpyHostToDevice), "Could not copy deviceParameter_Tset_spike_TestData2 to device");

			cudasafe(cudaMemcpy(&deviceParameter_Tset[j].spike_TestData, &temp_spike_TestData, sizeof(float*), cudaMemcpyHostToDevice), "Could not copy deviceParameter_Tset_spike_TestData to device");
			cudasafe(cudaFree(temp_spike_TestData), "Could not free temp_spike_TestData");
	
			
		}
		GAKernel_GenEach << <BLOCKS_NUMBER, THREADS_PER_BLOCK >> >(devicePopulation, deviceScore, randomStates, deviceParameter_, deviceParameter_Tset, tau, k, MaxGeneration);
		//delete[]population;
		//float *population = new float[POPULATION_SIZE * VAR_NUMBER];
		cudasafe(cudaMemcpy(population, devicePopulation, POPULATION_SIZE * VAR_NUMBER * sizeof(float), cudaMemcpyDeviceToHost), "Could not copy population from device");

		printf("%d_1111\n", k);
		printFinalPopulation(devicePopulation, deviceScore);
		printf("%d_2222\n", k);
		////GAKernel_gen << <BLOCKS_NUMBER, THREADS_PER_BLOCK >> >(devicePopulation, sharedPopulation, sharedScore, deviceScore, randomStates, deviceParameter_, deviceParameter_Tset, tau);
	}
	cudasafe(cudaGetLastError(), "Could not invoke GAKernel");
    cudasafe(cudaDeviceSynchronize(), "Failed to syncrhonize device after calling GAKernel");

    //printPopulation(devicePopulation, deviceScore);
    //}

	// freeing memory
	cudasafe(cudaFree(devicePopulation), "Failed to free devicePopulation");
	cudasafe(cudaFree(deviceScore), "Failed to free deviceScore");
	cudasafe(cudaFree(randomStates), "Could not free randomStates");
	cudasafe(cudaFree(nextGeneration), "Could not free nextGeneration");
	cudasafe(cudaFree(deviceParameter_Tset), "Could not free deviceParameter_Tset");
	delete[] population;

	return ans;
}

float * Read_Txt(string filename,int &num)
{
	float *Mdata = new float[100000];
	ifstream in(filename);
	string line;
	int i = 0;
	if (in) // 有该文件  
	{
		while (getline(in, line)) // line中不包括每行的换行符  
		{
			//cout << stringToNum<float>(line)+0.015 << endl;
			Mdata[i] = stringToNum<float>(line);
			i++;
		}
	}
	else // 没有该文件  
	{
		cout << "no such file" << endl;
		return 0;
	}
	num = i;
	float *Mdata_copy = new float[i];
	for (int j = 0; j < i; j++)
	{
		Mdata_copy[j] = Mdata[j];
	}
	return Mdata_copy;
}

int main() {
	freopen("output.txt", "w", stdout);
	srand(1900);
	srand(static_cast<unsigned>(time(0)));
	//float *spike_data, *current_data, *spike_TestData;
	M_args Parameter_;
	Parameter_.spike_data = Read_Txt("spikes.txt", Parameter_.spike_data_num);
	Parameter_.current_data = Read_Txt("current.txt", Parameter_.current_data_num);
	//Parameter_.spike_TestData = Read_Txt("spikes_test.txt");
	double ans = solveGPU(Parameter_);
	std::cout << "GPU answer = " << ans << std::endl;

	return 0;
}
