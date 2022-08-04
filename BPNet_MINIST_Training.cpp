
#include <iostream>
#include <fstream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cmath>
#include <string.h>

using namespace std;

const int input_node_num = 784 ;    //输入节点数量
const int hidden_node_num = 100 ;   //隐层节点数量
const int output_node_num = 10 ;    //输出节点数量
const double learning_rate = 0.35 ; //学习率

int input_layer[input_node_num] ;   //输入层
double hidden_layer_output[hidden_node_num] ;   //隐藏层
int output_layer_target[output_node_num] ;  //输出层正确参考数据
double output_layer_real[output_node_num] ; //输出层真实输出数据


double input_hidden_weight[input_node_num][hidden_node_num] ;   //输入层 -> 隐藏层 权重
double hidden_output_weight[hidden_node_num][output_node_num] ; //隐藏层 -> 输出层 权重

double hidden_bias[hidden_node_num] ;   //隐藏层 偏置
double output_bias[output_node_num] ;   //输出层 偏置

//计算delta改变量，可以减少运算次数，优化运行效率
double hidden_delta[hidden_node_num] ;  //隐藏层求变化量时须用到的delta偏导值
double output_delta[output_node_num] ;  //输出层求变化量时须用到的delta偏导值

void show_pic_info(int tag_value)   //显示图片信息，tag_value:正确数据
{
    cout<<"图像信息："<<endl;
    for(int i = 0; i < 784; i++)
        {
            if(input_layer[i]==1)
            {
                cout<<input_layer[i]<<",";
            }
			else
            {
                cout<<"  ";
			}
		}
    cout<<endl;
    cout<<"真实数值"<<tag_value<<endl<<endl;
}
void show_pic_info(int tag_value,int out_value)  //显示图片信息，tag_value:正确数据，out_value:真实输出数据
{
    cout<<"图像信息："<<endl;
    for(int i = 0; i < 784; i++)
        {
            if(input_layer[i]==1)
            {
                cout<<input_layer[i]<<",";
            }
			else
            {
                cout<<"  ";
			}
		}
    cout<<endl;
    cout<<"标签数值："<<tag_value<<"\t  实际输出："<<out_value<<endl<<endl;
}
double sigmod(double x)     //激活函数
{
    return 1.0 / (1.0 + exp(-x));
}
void forward_propagation()  //前向传递
{
    for(int i = 0; i < hidden_node_num; i++)
    {
        double sigmod_sum = 0.0 ;
        for(int j = 0; j < input_node_num; j++)
        {
            sigmod_sum += input_layer[j] * input_hidden_weight[j][i];
        }
        double x = sigmod_sum + hidden_bias[i];
        hidden_layer_output[i] = sigmod(x);
    }

    for(int i = 0; i < output_node_num; i++)
    {
        double sigmod_sum = 0.0 ;
        for(int j = 0; j < hidden_node_num; j++)
        {
            sigmod_sum += hidden_layer_output[j] * hidden_output_weight[j][i] ;
        }
        double x = sigmod_sum + output_bias[i];
        output_layer_real[i] = sigmod(x) ;
    }
}
void calcu_delta()  //计算delta求偏导的值
{
    //计算输出层delta
    for(int i = 0; i < output_node_num; i++)
    {
        output_delta[i] = (output_layer_real[i] - output_layer_target[i]) * (output_layer_real[i]) * (1.0 - output_layer_real[i]);
    }
    //计算隐藏层delta
    for(int i = 0; i < hidden_node_num; i++)
    {
        double delta_sum = 0;
        for(int j = 0; j < output_node_num; j++)
        {
            delta_sum += hidden_output_weight[i][j] * output_delta[j] ;
        }
        hidden_delta[i] = delta_sum * hidden_layer_output[i] * (1 - hidden_layer_output[i]);
    }
}
void backward_propagation_use_delta()   //反向传递，使用计算保存的delta值
{
    //更新输入层权重和隐层偏置
    for(int i = 0; i < hidden_node_num; i++)
    {
        hidden_bias[i] =  hidden_bias[i] - learning_rate * hidden_delta[i];
        for(int j = 0; j < input_node_num; j++)
        {
            input_hidden_weight[j][i] =  input_hidden_weight[j][i] - learning_rate * hidden_delta[i] * input_layer[j];
        }
    }
    //更新隐层权重和输出层偏置
    for(int i = 0; i < output_node_num ; i++)
    {
        output_bias[i] = output_bias[i] - learning_rate * output_delta[i];
        for(int j = 0; j < hidden_node_num; j++)
        {
            hidden_output_weight[j][i] =  hidden_output_weight[j][i] - learning_rate * output_delta[i] * hidden_layer_output[j];
        }
    }
}
void backward_propagation()     //反向传递，单独计算每个改变量，不使用delta值
{
    //更新输出层bias值
    for(int i = 0; i < output_node_num; i++)
    {
        double delta_bias = 0.0 ;
        delta_bias = (output_layer_real[i] - output_layer_target[i]) * output_layer_real[i] * (1.0 - output_layer_real[i]);
        output_bias[i] = output_bias[i] - (learning_rate * delta_bias);
    }
    //更新 隐藏 -> 输出层 权重
    for(int i = 0; i < hidden_node_num; i++)
    {
        for(int j = 0; j < output_node_num; j++)
        {
           hidden_output_weight[i][j] =  hidden_output_weight[i][j] - (learning_rate * (output_layer_real[j] - output_layer_target[j]) * output_layer_real[j] * (1 - output_layer_real[j]) * hidden_layer_output[i]);
        }
    }
    //更新隐藏层bias值
    for(int i = 0; i < hidden_node_num; i++)
    {
        double delta_sum = 0.0 ;
        for(int j = 0 ;j < output_node_num; j++)
        {
            delta_sum = delta_sum + (output_layer_real[j] - output_layer_target[j]) * output_layer_real[j] * (1 - output_layer_real[j]) * hidden_output_weight[i][j];
        }
        hidden_bias[i] = hidden_bias[i] - learning_rate * delta_sum * hidden_layer_output[i] * (1 - hidden_layer_output[i]);
    }
    //更新 输入 -> 隐藏 权重值
    for(int i = 0; i < input_node_num; i++)
    {
        double delta_sum = 0.0 ;
        for(int j = 0; j < hidden_node_num; j++)
        {
            for(int k = 0; k < output_node_num; k++)
            {
                delta_sum = delta_sum + (output_layer_real[k] - output_layer_target[k]) * output_layer_real[k] * (1 - output_layer_real[k]) * hidden_output_weight[j][k];
            }
            delta_sum = delta_sum * hidden_layer_output[j] * (1 - hidden_layer_output[j]) * input_layer[i];
            input_hidden_weight[i][j] = input_hidden_weight[i][j] - learning_rate * delta_sum;
        }
    }
}
double get_randnum()    //获得随机0-1数
{
    return (rand() % 1000 * 0.001 - 0.5);
}
double get_random(int min,int max)      //获得指定随机数
{
     return ( rand() % (max - min + 1) ) + min ;
}
void initialize_bpnet()     //初始化网络
{
    srand((int)time(0) + rand());
    for(int i = 0; i < input_node_num; i++)
    {
        for(int j = 0; j < hidden_node_num; j++)
        {
            input_hidden_weight[i][j] = get_randnum() ;
        }
    }

    for(int i = 0; i < hidden_node_num; i++)
    {
        for(int j = 0; j < output_node_num ; j++)
        {
            hidden_output_weight[i][j] = get_randnum();
        }
    }

    for(int i = 0; i < hidden_node_num ; i++)
    {
        hidden_bias[i] = get_randnum();
    }

    for(int i = 0; i < hidden_node_num ; i++)
    {
        output_bias[i] = get_randnum();
    }
}
void training() //开始训练
{
    FILE* training_image;
    FILE* training_label;
    int training_count = 0;
    unsigned char image_buffer[784];    //保存图片信息
    unsigned char label_buffer[10];     //保存标签数据

    training_image = fopen("./train-images.idx3-ubyte", "rb");
    training_label = fopen("./train-labels.idx1-ubyte", "rb");
    if (training_image == NULL || training_label == NULL) {
		cout << "open training file error" << endl;
		exit(0);
	}

	int head_info[1000];    //读取出文件的头信息，该信息不参与计算
	fread(head_info,1,16,training_image);   //读取16字节头部信息
	fread(head_info,1,8,training_label);    //读取8字节头部信息

	cout<<"Training started..."<<endl;

	while(!feof(training_image) && !feof(training_label))
    {
        memset(image_buffer,0,784);
        memset(label_buffer,0,10);
        fread(image_buffer,1,784,training_image);   //读取一个图片
        fread(label_buffer,1,1,training_label);     //读取一个标签

        for(int i = 0; i < 784; i++)    //判断灰度值为0或者为1，初始化输入层数据
        {
            if((unsigned int)image_buffer[i] < 128)
            {
                input_layer[i] = 0;
            }
            else
            {
                input_layer[i] = 1;
            }
        }

        int label_value = (unsigned int)label_buffer[0];    //获得标签数据

        //show_pic_info(label_value);

        for(int i = 0; i < output_node_num; i++)
        {
            output_layer_target[i] = 0 ;
        }
        output_layer_target[label_value] = 1;   //设置输出节点的正确数据

        forward_propagation();  //前向传递
        calcu_delta();
        backward_propagation_use_delta();
        //backward_propagation(); //反向传递

        training_count++;
        if (training_count % 1500 == 0)
        {
             cout << "训练程度: " << (training_count / 60000.0) * 100.0<<"%"<< endl;
        }
    }
     cout<<"训练完毕！！！"<<endl;
}
void testing()  //测试
{
    FILE *testing_image;
	FILE *testing_label;
	testing_image = fopen("./t10k-images.idx3-ubyte", "rb");
	testing_label = fopen("./t10k-labels.idx1-ubyte", "rb");

    double test_num = 0.0 ;
    double test_success_num = 0.0 ;

	if (testing_image == NULL || testing_label == NULL)
    {
		cout << "open training file error!" << endl;
		exit(0);
	}

	unsigned char image_buffer[784];
	unsigned char label_buffer[1];

	int useless[1000];
	fread(useless, 1, 16, testing_image);
	fread(useless, 1, 8, testing_label);

	while (!feof(testing_image) && !feof(testing_label))
    {
		memset(image_buffer, 0, 784);
		memset(label_buffer, 0, 1);
		fread(image_buffer, 1, 784, testing_image);
		fread(label_buffer, 1, 1, testing_label);

		for (int i = 0; i < 784; i++)
        {
			if ((unsigned int)image_buffer[i] < 128)
			{
				input_layer[i] = 0;
			}
			else
            {
				input_layer[i] = 1;
			}
		}

		for (int k = 0; k < output_node_num; k++)
        {
			output_layer_target[k] = 0;
		}

		int target_value = (unsigned int)label_buffer[0];
		output_layer_target[target_value] = 1;

        forward_propagation();

		double max_value = -99999;
		int max_index = 0;
		for (int k = 0; k < output_node_num; k++)   //寻找输出层节点最大的值得坐标
        {
			if (output_layer_real[k] > max_value)
			{
				max_value = output_layer_real[k];
				max_index = k;
			}
		}

		if (output_layer_target[max_index] == 1)    //判断正确输出的坐标位置是否一致
        {
			test_success_num ++;
		}

		test_num ++;

		if ((int)test_num % 1000 == 0)
        {
			cout << "测试数量: " << test_num << "  成功数量: " << test_success_num << endl;
		}
		if( test_num>=9995&&test_num<10000)
        {
             show_pic_info(target_value,max_index);
        }
	}
	cout << endl;
	cout << "成功率: " << test_success_num / test_num << endl;
}
int main()
{
    initialize_bpnet();
    training();
    testing();
    system("pause");
    return 0;
}
