const int VAR_NUMBER = 5;
const double KNOWN_ANSWER = 0;
const int POPULATION_SIZE = 1024*5; // should be a multiple of block size
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
	//int length;
	int spike_data_num;
	int current_data_num;
};
//function HH_return for test
float *HH_return(float *List_param, int VAR_NUMBER,int &num)
{
	float *Array_Data = new float[10];
	float *new_param = new float[5];
	for (int i = 0; i < 10; i++)
	{
		Array_Data[i] = List_param[0];
	}
	num = 10;
	return Array_Data;
}