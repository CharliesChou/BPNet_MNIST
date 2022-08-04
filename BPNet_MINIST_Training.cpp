
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

const int input_node_num = 784 ;    //����ڵ�����
const int hidden_node_num = 100 ;   //����ڵ�����
const int output_node_num = 10 ;    //����ڵ�����
const double learning_rate = 0.35 ; //ѧϰ��

int input_layer[input_node_num] ;   //�����
double hidden_layer_output[hidden_node_num] ;   //���ز�
int output_layer_target[output_node_num] ;  //�������ȷ�ο�����
double output_layer_real[output_node_num] ; //�������ʵ�������


double input_hidden_weight[input_node_num][hidden_node_num] ;   //����� -> ���ز� Ȩ��
double hidden_output_weight[hidden_node_num][output_node_num] ; //���ز� -> ����� Ȩ��

double hidden_bias[hidden_node_num] ;   //���ز� ƫ��
double output_bias[output_node_num] ;   //����� ƫ��

//����delta�ı��������Լ�������������Ż�����Ч��
double hidden_delta[hidden_node_num] ;  //���ز���仯��ʱ���õ���deltaƫ��ֵ
double output_delta[output_node_num] ;  //�������仯��ʱ���õ���deltaƫ��ֵ

void show_pic_info(int tag_value)   //��ʾͼƬ��Ϣ��tag_value:��ȷ����
{
    cout<<"ͼ����Ϣ��"<<endl;
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
    cout<<"��ʵ��ֵ"<<tag_value<<endl<<endl;
}
void show_pic_info(int tag_value,int out_value)  //��ʾͼƬ��Ϣ��tag_value:��ȷ���ݣ�out_value:��ʵ�������
{
    cout<<"ͼ����Ϣ��"<<endl;
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
    cout<<"��ǩ��ֵ��"<<tag_value<<"\t  ʵ�������"<<out_value<<endl<<endl;
}
double sigmod(double x)     //�����
{
    return 1.0 / (1.0 + exp(-x));
}
void forward_propagation()  //ǰ�򴫵�
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
void calcu_delta()  //����delta��ƫ����ֵ
{
    //���������delta
    for(int i = 0; i < output_node_num; i++)
    {
        output_delta[i] = (output_layer_real[i] - output_layer_target[i]) * (output_layer_real[i]) * (1.0 - output_layer_real[i]);
    }
    //�������ز�delta
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
void backward_propagation_use_delta()   //���򴫵ݣ�ʹ�ü��㱣���deltaֵ
{
    //���������Ȩ�غ�����ƫ��
    for(int i = 0; i < hidden_node_num; i++)
    {
        hidden_bias[i] =  hidden_bias[i] - learning_rate * hidden_delta[i];
        for(int j = 0; j < input_node_num; j++)
        {
            input_hidden_weight[j][i] =  input_hidden_weight[j][i] - learning_rate * hidden_delta[i] * input_layer[j];
        }
    }
    //��������Ȩ�غ������ƫ��
    for(int i = 0; i < output_node_num ; i++)
    {
        output_bias[i] = output_bias[i] - learning_rate * output_delta[i];
        for(int j = 0; j < hidden_node_num; j++)
        {
            hidden_output_weight[j][i] =  hidden_output_weight[j][i] - learning_rate * output_delta[i] * hidden_layer_output[j];
        }
    }
}
void backward_propagation()     //���򴫵ݣ���������ÿ���ı�������ʹ��deltaֵ
{
    //���������biasֵ
    for(int i = 0; i < output_node_num; i++)
    {
        double delta_bias = 0.0 ;
        delta_bias = (output_layer_real[i] - output_layer_target[i]) * output_layer_real[i] * (1.0 - output_layer_real[i]);
        output_bias[i] = output_bias[i] - (learning_rate * delta_bias);
    }
    //���� ���� -> ����� Ȩ��
    for(int i = 0; i < hidden_node_num; i++)
    {
        for(int j = 0; j < output_node_num; j++)
        {
           hidden_output_weight[i][j] =  hidden_output_weight[i][j] - (learning_rate * (output_layer_real[j] - output_layer_target[j]) * output_layer_real[j] * (1 - output_layer_real[j]) * hidden_layer_output[i]);
        }
    }
    //�������ز�biasֵ
    for(int i = 0; i < hidden_node_num; i++)
    {
        double delta_sum = 0.0 ;
        for(int j = 0 ;j < output_node_num; j++)
        {
            delta_sum = delta_sum + (output_layer_real[j] - output_layer_target[j]) * output_layer_real[j] * (1 - output_layer_real[j]) * hidden_output_weight[i][j];
        }
        hidden_bias[i] = hidden_bias[i] - learning_rate * delta_sum * hidden_layer_output[i] * (1 - hidden_layer_output[i]);
    }
    //���� ���� -> ���� Ȩ��ֵ
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
double get_randnum()    //������0-1��
{
    return (rand() % 1000 * 0.001 - 0.5);
}
double get_random(int min,int max)      //���ָ�������
{
     return ( rand() % (max - min + 1) ) + min ;
}
void initialize_bpnet()     //��ʼ������
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
void training() //��ʼѵ��
{
    FILE* training_image;
    FILE* training_label;
    int training_count = 0;
    unsigned char image_buffer[784];    //����ͼƬ��Ϣ
    unsigned char label_buffer[10];     //�����ǩ����

    training_image = fopen("./train-images.idx3-ubyte", "rb");
    training_label = fopen("./train-labels.idx1-ubyte", "rb");
    if (training_image == NULL || training_label == NULL) {
		cout << "open training file error" << endl;
		exit(0);
	}

	int head_info[1000];    //��ȡ���ļ���ͷ��Ϣ������Ϣ���������
	fread(head_info,1,16,training_image);   //��ȡ16�ֽ�ͷ����Ϣ
	fread(head_info,1,8,training_label);    //��ȡ8�ֽ�ͷ����Ϣ

	cout<<"Training started..."<<endl;

	while(!feof(training_image) && !feof(training_label))
    {
        memset(image_buffer,0,784);
        memset(label_buffer,0,10);
        fread(image_buffer,1,784,training_image);   //��ȡһ��ͼƬ
        fread(label_buffer,1,1,training_label);     //��ȡһ����ǩ

        for(int i = 0; i < 784; i++)    //�жϻҶ�ֵΪ0����Ϊ1����ʼ�����������
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

        int label_value = (unsigned int)label_buffer[0];    //��ñ�ǩ����

        //show_pic_info(label_value);

        for(int i = 0; i < output_node_num; i++)
        {
            output_layer_target[i] = 0 ;
        }
        output_layer_target[label_value] = 1;   //��������ڵ����ȷ����

        forward_propagation();  //ǰ�򴫵�
        calcu_delta();
        backward_propagation_use_delta();
        //backward_propagation(); //���򴫵�

        training_count++;
        if (training_count % 1500 == 0)
        {
             cout << "ѵ���̶�: " << (training_count / 60000.0) * 100.0<<"%"<< endl;
        }
    }
     cout<<"ѵ����ϣ�����"<<endl;
}
void testing()  //����
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
		for (int k = 0; k < output_node_num; k++)   //Ѱ�������ڵ�����ֵ������
        {
			if (output_layer_real[k] > max_value)
			{
				max_value = output_layer_real[k];
				max_index = k;
			}
		}

		if (output_layer_target[max_index] == 1)    //�ж���ȷ���������λ���Ƿ�һ��
        {
			test_success_num ++;
		}

		test_num ++;

		if ((int)test_num % 1000 == 0)
        {
			cout << "��������: " << test_num << "  �ɹ�����: " << test_success_num << endl;
		}
		if( test_num>=9995&&test_num<10000)
        {
             show_pic_info(target_value,max_index);
        }
	}
	cout << endl;
	cout << "�ɹ���: " << test_success_num / test_num << endl;
}
int main()
{
    initialize_bpnet();
    training();
    testing();
    system("pause");
    return 0;
}
