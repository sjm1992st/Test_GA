const int VAR_NUMBER = 5;
const double KNOWN_ANSWER = 0;
const int POPULATION_SIZE = 1024*6 + 512; // should be a multiple of block size
const unsigned U_RAND_MAX = static_cast<unsigned>(RAND_MAX) + 1;

// random number in [0, 1)
float float_random() {
	return (static_cast<float>(rand())) / (U_RAND_MAX);
}

struct ScoreWithId {
	float score;
	int id;
};

//struct HH_Param{
//	float Na_0 = 0;
//	float K_0 = 0;
//	float Ca_0 = 0;
//
//};
struct M_args_Tset
{
	float *spike_TestData;
	int length;
};

struct M_args
{
	float *spike_data;
	float *current_data;
	//float *spike_TestData;
	float dt;
	int length;
};
//function HH_return for test
float *HH_return(float *List_param, int VAR_NUMBER)
{
	float *Array_Data = new float[10000];
	float *new_param = new float[5];
	for (int i = 0; i < 10000; i++)
	{
		Array_Data[i] = List_param[0];
	}
	return Array_Data;
}