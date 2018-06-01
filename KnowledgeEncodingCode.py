from random import seed
from random import randrange
import random
from csv import reader
from math import exp
from sklearn.metrics import confusion_matrix
import numpy as np
import math
import itertools
from collections import Counter

def load_csv(filename):
        dataset = list()
        with open(filename, 'r') as file:
                csv_reader = reader(file)
                for row in csv_reader:
                        if not row:
                                continue
                        dataset.append(row)
        return dataset
 
# Convert string column to float
def str_column_to_float(dataset, column):
        for row in dataset:
                try:
                        row[column] = float(row[column].strip())
                except ValueError:
                        print("Error with row",column,":",row[column])
                        pass
 
# Convert string column to integer
def str_column_to_int(dataset, column):        
        for row in dataset:
                row[column] = int(row[column])          
                                
def splitData(dataset, sRatio):
        trainSize = int(len(dataset) * sRatio)
        trainSet = []
        copy = list(dataset)
        seed(8)
        while len(trainSet) < trainSize:                
                index = random.randrange(len(copy))
                
                trainSet.append(copy.pop(index))
        return [trainSet, copy]

def generate_new_dataset(new_sam,row):
        #print(row)       
        X = np.empty((len(new_sam), 0))
        #print(X)
        for i in range(len(row)):
                col=row[i]
                x=[row_new[col-1] for row_new in new_sam]
                
                x=np.array([x])
                #print(np.transpose(x))
                #print(x)
                x=x.T
                X=np.append(X, x, axis=1)
        x=[row[-1] for row in new_sam]
        x=np.array([x])
        x=x.T
        X=np.append(X, x, axis=1)
        X=np.array(X).tolist()
        #print("X {}".format(X))
        return X

def fuzzify(trainingSet, testSet):
        center=list()
        rad=list()
        for i in range(3):
                feature1=list()
                feature1=[row[i] for row in trainingSet]
                #print("feature {}".format(feature1))
                maxi=max(feature1)
                mini=min(feature1)
                
                radius_medium=0.5*(maxi-mini)
                #print("rad {}".format(radius_medium))
                center_medium= mini + radius_medium
                radius_low= 2*(center_medium-mini)
                center_low=center_medium-(0.5*radius_low)
                radius_high = 2*(maxi - center_medium)
                center_high = center_medium + (0.5 * radius_high)
                center.append([center_low,center_medium,center_high])
                rad.append([radius_low,radius_medium,radius_high])

        fuzzy=list()
        for i in range(len(trainingSet)):
            data=trainingSet[i]
            #print(data)
            l=0
            value=list()
            for j in range(3):
                d=data[j]        
                for k in range(3):
                    r=rad[j][k]/2
                    eucleanD=math.pow((d-center[j][k]),2)
                    eDist=math.sqrt(eucleanD)           
                    if eDist <= r :
                        val = 1- 2*math.pow(eDist/(r*2),2)
                        value.insert(l,val)
                    else:
                        y=eDist/rad[j][k]
                        x=(1-(eDist/rad[j][k]))
                        val = 2*math.pow(x,2)
                        value.insert(l,val)
                    l=l+1
            value.insert(l,data[-1])
            fuzzy.insert(i,value)
        #print(fuzzy)
        fuzzy_test=list()
        for i in range(len(testSet)):
            data=testSet[i]
            #print(data)
            l=0
            value=list()
            for j in range(3):
                d=data[j]        
                for k in range(3):
                    r=rad[j][k]/2
                    eucleanD=math.pow((d-center[j][k]),2)
                    eDist=math.sqrt(eucleanD)           
                    if eDist <= r :
                        val = 1- 2*math.pow(eDist/(r*2),2)
                        value.insert(l,val)
                    else:
                        y=eDist/rad[j][k]
                        x=(1-(eDist/rad[j][k]))
                        val = 2*math.pow(x,2)
                        value.insert(l,val)
                    l=l+1
            value.insert(l,data[-1])
            fuzzy_test.insert(i,value)
            
        fuzzy_threshold=list()
        for i in range(len(trainingSet)):
                row=list()
                temp=0
                data= fuzzy[i]
                for j in range(len(data)-1):
                        d=data[j]
                        if d < 0.8:
                                temp=0
                        else:
                                temp=1
                        row.append(temp)
                row.append(data[-1])
                fuzzy_threshold.append(row)
        #print(fuzzy_threshold, len(fuzzy_threshold))
        return [fuzzy_threshold, fuzzy, fuzzy_test]
        
def process(trainingSet,fuzzy_threshold):
        folds=list()
        for j in range(6):        
                fold=list()
                for i in range(len(trainingSet)):
                        data=fuzzy_threshold[i]
                        if data[-1]==j+1:
                                fold.append(fuzzy_threshold[i])
                folds.append(fold)
        #print(folds)
        final_feature=list()
        for fold in folds:
                fold.sort()
                #print("    {} \n\n".format(fold))
                output=set(tuple(i) for i in fold)
                tup=(list(o) for o in output)
                largest_count=0
                for ele in tup:
                        #print(ele)
                        count=0
                        for i in range(len(fold)):
                                ele_fold=fold[i]
                                compare = lambda a,b: len(a)==len(b) and len(a)==sum([1 for i,j in zip(a,b) if i==j])
                                if compare(ele, ele_fold):
                                        count=count+1
                        if count > largest_count:
                                largest_count=count
                                rep_feature=ele
                final_feature.append(rep_feature)
        #for i in range(6):
                #print(final_feature[i])
        return final_feature

def dependency(new_data,num):
        count=len(new_data)
        dependency=0
        for row in new_data:
                del row[-1]

        no_dupes = [x for n, x in enumerate(new_data) if x in new_data[:n]]
        #print(no_dupes)

        for i in range(len(no_dupes)):
                while no_dupes[i] in new_data:
                        new_data.remove(no_dupes[i])

        #print(new_data)
        dependency=len(new_data)/count
        return dependency
        '''for j in range(6):
                k=0
                while k<6:
                        if j!= k: 
                                list1=new_data[j]                  
                                list2=new_data[k]
                                if list1[:num]==list2[:num] and list1[-1]!=list2[-1]:                                        
                                        count = count-1
                                        print("Count inside {}".format(count))
                        k=k+1                        
        print("count {}",format(count))
        dependency=count/len(new_data)
        print(" {} ".format(dependency))
        return dependency'''

def compute_minterm(sample_set):
        total_minterm=list()
        for i in range(6):
                j=i+1
                temp1 = sample_set[i]
                #print(len(temp1))
                while (j<6):
                        temp2 = sample_set[j]
                        #print("1 2 {} {}".format(temp1, temp2))
                        minterm=list()
                        for k in range(len(temp1)-1):                        
                                if temp1[k] != temp2[k]:
                                        minterm.insert(k,k+1)                                
                        #print(" minterm{}".format(minterm))
                        total_minterm.insert(i,minterm)
                        j=j+1

        #print(total_minterm)
        s=list()
        #this part of the code removes duplicate entries
        for i in total_minterm:
                if i not in s:
                        s.append(i) 

        ##this part of the code removes null set
        s1 = [x for x in s if x]

        #this part of the code captures all minterms that should be removed because of the absorption law
        #The minterms are collected in 'rem'
        s1.sort(key=len)
        #print("\n\n {}".format(s1))
        rem=list()
        for i in range(len(s1)):
                one=s1[i]
                #print("one{}".format(one))
                j=i+1
                while j<len(s1):
                        two=s1[j]
                        #print("j={} two{}".format(j,two))
                        if all(x in two for x in one) and two not in rem:
                                rem.append(two)
                                #print(rem)
                        j=j+1

        #The minterms are now removed
        for item in rem:
                s1.remove(item)
        return s1

def minterm_for_rules(sample_set,x):
        mint=list()
        inp_neuron_count=0
        dict_weight={}
        #hidden=[]
        dict_wei_output={}
       
        #output_layer=[]
        count_n=0
        for i in range(6):
                total_minterm=list()
                j=0
                temp1 = sample_set[i]
                print(len(temp1))
                while (j<6):
                        temp2 = sample_set[j]
                        #print(" {} {}".format(temp1, temp2))
                        minterm=list()
                        for k in range(len(temp1)-1):                        
                                if temp1[k] != temp2[k]:
                                        minterm.insert(k,x[k])                                
                        #print(" minterm{}".format(minterm))
                        total_minterm.insert(i,minterm)
                        j=j+1
                s=list()
                #this part of the code removes duplicate entries
                for p in total_minterm:
                        if p not in s:
                                s.append(p)
                s1 = [x for x in s if x]
                #this part of the code captures all minterms that should be removed because of the absorption law
                #The minterms are collected in 'rem'
                s1.sort(key=len)
                #print("\n\n {}".format(s1))
                rem=list()
                for w in range(len(s1)):
                        one=s1[w]
                        #print("one{}".format(one))
                        j=w+1
                        while j<len(s1):
                                two=s1[j]
                                #print("j={} two{}".format(j,two))
                                if all(x in two for x in one) and two not in rem:
                                        rem.append(two)
                                        #print(rem)
                                j=j+1
                #The minterms are now removed
                for item in rem:
                        s1.remove(item)
                #print(s1)
                
                inp_neuron_count=inp_neuron_count+len(s1)
                if i==0:
                        dep_one=0
                        class_one.append(s1)
                        for j in itertools.product(*s1):
                                #print(i)
                                new_s=generate_new_dataset(final_sample,j)
                                dep=dependency(new_s,len(j))
                                dep_one=dep_one+dep
                        #print("dep_one {}".format(dep_one))
                        wei=dep_one/len(s1)
                        print(final_sample[i])
                        wt_output=[0 for i in range(12)]
                        for p in range(len(s1)):
                                wt=[0 for i in range(9)]
                                
                                for j in range(len(s1[p])):
                                        pos=s1[p][j]
                                        if final_sample[i][pos-1] ==1:
                                                wt[pos-1]=wei/len(s1[p])
                                        else:
                                                wt[pos-1]=-wei/len(s1[p])                                                
                                        dict_weight['weights']=wt
                                
                                wt_output[count_n]=wei                                
                                count_n=count_n+1                                                                               
                                hidden_layer.append(dict_weight)
                                dict_weight={}                                
                                #dict_wei_output={}
                        dict_wei_output['weights']=wt_output
                        output_layer.append(dict_wei_output)
                        dict_wei_output={}
                        #print(hidden_layer)
                        #print("OUTPUT {}".format(output_layer))
                        
                elif i==1:
                        dep_two=0
                        class_two.append(s1)
                        for j in itertools.product(*s1):
                                new_s=generate_new_dataset(final_sample,j)
                                dep=dependency(new_s,len(j))
                                dep_two=dep_two+dep
                        #print("dep_two {}".format(dep_two))
                        wei=dep_two/len(s1)
                        wt_output=[0 for i in range(12)]
                        for p in range(len(s1)):
                                wt=[0 for i in range(9)]
                                
                                for j in range(len(s1[p])):
                                        pos=s1[p][j]                                                                                
                                        if final_sample[i][pos-1] ==1:
                                                wt[pos-1]=wei/len(s1[p])
                                        else:
                                                wt[pos-1]=-wei/len(s1[p])                                        
                                        dict_weight['weights']=wt
                                        
                                wt_output[count_n]=wei                                
                                count_n=count_n+1                                        
                                hidden_layer.append(dict_weight)
                                dict_weight={}                              
                                
                        dict_wei_output['weights']=wt_output
                        output_layer.append(dict_wei_output)
                        dict_wei_output={}
                        #print(hidden_layer)
                        #print("OUTPUT {}".format(output_layer))
                        
                elif i==2:
                        dep_three=0
                        class_three.append(s1)
                        for j in itertools.product(*s1):
                                new_s=generate_new_dataset(final_sample,j)
                                dep=dependency(new_s,len(j))
                                dep_three=dep_three+dep
                        #print("dep_three {}".format(dep_three))
                        wei=dep_three/len(s1)
                        wt_output=[0 for i in range(12)]
                        for p in range(len(s1)):
                                wt=[0 for i in range(9)]
                                wt_output=[0 for i in range(12)]
                                for j in range(len(s1[p])):
                                        pos=s1[p][j]                                                                                
                                        if final_sample[i][pos-1] ==1:
                                                wt[pos-1]=wei/len(s1[p])
                                        else:
                                                wt[pos-1]=-wei/len(s1[p])                                        
                                        dict_weight['weights']=wt
                                
                                wt_output[count_n]=wei                                
                                count_n=count_n+1                                        
                                hidden_layer.append(dict_weight)
                                dict_weight={}                              
                                
                        dict_wei_output['weights']=wt_output
                        output_layer.append(dict_wei_output)
                        dict_wei_output={}
                        #print(hidden_layer)
                        #print("OUTPUT {}".format(output_layer))
                        
                elif i==3:
                        dep_four=0
                        class_four.append(s1)
                        for j in itertools.product(*s1):
                                new_s=generate_new_dataset(final_sample,j)
                                dep=dependency(new_s,len(j))
                                dep_four=dep_four+dep
                        #print("dep_four {}".format(dep_four))
                        wei=dep_four/len(s1)
                        wt_output=[0 for i in range(12)]
                        for p in range(len(s1)):
                                wt=[0 for i in range(9)]
                                
                                for j in range(len(s1[p])):
                                        pos=s1[p][j]                                                                                
                                        if final_sample[i][pos-1] ==1:
                                                wt[pos-1]=wei/len(s1[p])
                                        else:
                                                wt[pos-1]=-wei/len(s1[p])                                        
                                        dict_weight['weights']=wt
                                
                                wt_output[count_n]=wei                                
                                count_n=count_n+1                                        
                                hidden_layer.append(dict_weight)
                                dict_weight={}
                                
                        dict_wei_output['weights']=wt_output
                        output_layer.append(dict_wei_output)
                        dict_wei_output={}
                        #print(hidden_layer)
                        #print("OUTPUT {}".format(output_layer))
                        
                elif i==4:
                        dep_five=0
                        class_five.append(s1)
                        for j in itertools.product(*s1):
                                new_s=generate_new_dataset(final_sample,j)
                                dep=dependency(new_s,len(j))
                                dep_five=dep_five+dep
                        #print("dep_five {}".format(dep_five))
                        wei=dep_five/len(s1)
                        wt_output=[0 for i in range(12)]
                        for p in range(len(s1)):
                                wt=[0 for i in range(9)]
                                
                                for j in range(len(s1[p])):
                                        pos=s1[p][j]                                                                                
                                        if final_sample[i][pos-1] ==1:
                                                wt[pos-1]=wei/len(s1[p])
                                        else:
                                                wt[pos-1]=-wei/len(s1[p])                                        
                                        dict_weight['weights']=wt                                
                                wt_output[count_n]=wei                                
                                count_n=count_n+1                                        
                                hidden_layer.append(dict_weight)
                                dict_weight={}                                
                        dict_wei_output['weights']=wt_output
                        output_layer.append(dict_wei_output)
                        dict_wei_output={}
                        #print(hidden_layer)
                        #print("OUTPUT {}".format(output_layer))
                        
                elif i==5:
                        dep_six=0
                        class_six.append(s1)
                        for j in itertools.product(*s1):
                                new_s=generate_new_dataset(final_sample,j)
                                dep=dependency(new_s,len(j))
                                dep_six=dep_six+dep                        
                        #print("dep_six {}".format(dep_six))
                        wei=dep_six/len(s1)
                        wt_output=[0 for i in range(12)]
                        for p in range(len(s1)):
                                wt=[0 for i in range(9)]
                                
                                for j in range(len(s1[p])):
                                        pos=s1[p][j]                                                                                
                                        if final_sample[i][pos-1] ==1:
                                                wt[pos-1]=wei/len(s1[p])
                                        else:
                                                wt[pos-1]=-wei/len(s1[p])                                        
                                        dict_weight['weights']=wt
                                
                                wt_output[count_n]=wei                                
                                count_n=count_n+1                                        
                                hidden_layer.append(dict_weight)
                                dict_weight={}                                
                        dict_wei_output['weights']=wt_output
                        output_layer.append(dict_wei_output)
                        dict_wei_output={}
                        #print(hidden_layer)
                        #print("OUTPUT {}".format(output_layer))
                        
        #print(class_one)
        #print(class_two)
        #print(class_three)
        #print(class_four)
        #print(class_five)
        #print(class_six)
        #print(inp_neuron_count)
        return inp_neuron_count



def accuracy_met(actual, predicted):
        correct = 0
        for i in range(len(actual)):
                if actual[i] == predicted[i]:
                        correct += 1
        return correct / float(len(actual)) * 100.0
 
# Evaluate an algorithm using a cross validation split
def run_algorithm(fuzzy, testSet, algorithm, *args):
        #print(dataset)
        predicted = algorithm(fuzzy, testSet, *args)
        actual = [row[-1] for row in testSet]
        #print(predicted)
        #print(actual)
        accuracy = accuracy_met(actual, predicted)
        cm = confusion_matrix(actual, predicted)
        print('\n'.join([''.join(['{:4}'.format(item) for item in row]) for row in cm]))
        #confusionmatrix = np.matrix(cm)
        FP = cm.sum(axis=0) - np.diag(cm)
        FN = cm.sum(axis=1) - np.diag(cm)
        TP = np.diag(cm)
        TN = cm.sum() - (FP + FN + TP)
        print('False Positives\n {}'.format(FP))
        print('False Negetives\n {}'.format(FN))
        print('True Positives\n {}'.format(TP))
        print('True Negetives\n {}'.format(TN))
        TPR = TP/(TP+FN)
        print('Sensitivity \n {}'.format(TPR))
        TNR = TN/(TN+FP)
        print('Specificity \n {}'.format(TNR))
        Precision = TP/(TP+FP)
        print('Precision \n {}'.format(Precision))
        Recall = TP/(TP+FN)
        print('Recall \n {}'.format(Recall))
        Acc = (TP+TN)/(TP+TN+FP+FN)
        print('Áccuracy \n{}'.format(Acc))
        Fscore = 2*(Precision*Recall)/(Precision+Recall)
        print('FScore \n{}'.format(Fscore))
        
        
                
        
 
# Calculate neuron activation for an input
def activate(weights, inputs):
        #print("weight neuorn {} {}".format(weights,inputs))
        activation = weights[-1]
        for i in range(len(weights)-1):
                activation += weights[i] * inputs[i]
        return activation
 
# Transfer neuron activation
def function(activation):
        return 1.0 / (1.0 + exp(-activation))
 
# Forward propagate input to a network output
def forward_propagate(network, row):
        inputs = row
        #print("input row{}\n".format(inputs))
        for layer in network:
                new_inputs = []
                for neuron in layer:
                        activation = activate(neuron['weights'], inputs)
                        neuron['output'] = function(activation)
                        new_inputs.append(neuron['output'])
                inputs = new_inputs
        #print("output row{}\n".format(inputs))
        return inputs
 
# Calculate the derivative of an neuron output
def function_derivative(output):
        return output * (1.0 - output)
 
# Backpropagate error and store in neurons
def backprop_error(network, expected):
        for i in reversed(range(len(network))):
                layer = network[i]
                errors = list()
                if i != len(network)-1:
                        for j in range(len(layer)):
                                error = 0.0
                                for neuron in network[i + 1]:
                                        error += (neuron['weights'][j] * neuron['delta'])
                                errors.append(error)
                else:
                        for j in range(len(layer)):
                                neuron = layer[j]
                                #print("neuron {} {}".format(j,neuron))
                                errors.append(expected[j] - neuron['output'])
                for j in range(len(layer)):
                        neuron = layer[j]
                        neuron['delta'] = errors[j] * function_derivative(neuron['output'])
 
# Update network weights with error
def change_weights(network, row, l_rate):
        for i in range(len(network)):
                inputs = row[:-1]
                if i != 0:
                        inputs = [neuron['output'] for neuron in network[i - 1]]
                for neuron in network[i]:
                        for j in range(len(inputs)):
                                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
                        neuron['weights'][-1] += l_rate * neuron['delta']

#To fuzzify the output layer
def fuzzyout(row,n_outputs):
        z=list()
        mu=list()
        muINT=list()
        rowclass=row[-1]-1
        #print(rowclass)
        for k in range(n_outputs):
                sumz=0
                for j in range(9):
                    interm=pow((row[j]-mean[k][j])/stdev[k][j],2)
                    sumz=sumz+interm
                    #print("row{}".format(row[j]))
                    #print("mean{}".format(mean[rowclass][j]))
                    #print("sum{}".format(sumz))
                weightedZ=math.sqrt(sumz)
                memMU=1/(1+(weightedZ/5))
                if 0 <= memMU <= 0.5:
                    memMUINT=2*pow(memMU,2)
                else:
                    temp=1-memMU
                    memMUINT=1-(2*pow(temp,2))
                mu.append(memMU)
                z.append(weightedZ)
                muINT.append(memMUINT)
        return muINT
 
# Train a network for a fixed number of epochs
def neural_network_train(network, train, l_rate, n_epoch, n_outputs):
        #print(dataset)
        for epoch in range(n_epoch):
                #print(train)
                for row in train:
                        #print("epochs {} {}".format(epoch,row))
                        outputs = forward_propagate(network, row)
                        #print(outputs)
                        expected = fuzzyout(row, n_outputs)
                        #print("input row{}\n".format(row))
                        #expected = [0 for i in range(n_outputs)]
                        #expected[row[-1]-1] = 1                        
                        #print("expected row{}\n".format(expected))
                        backprop_error(network, expected)
                        change_weights(network, row, l_rate)
 
# Initialize a network
def init_net(n_inputs, n_hidden, n_outputs):
        network = list()
        #hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
        print(hidden_layer)
        network.append(hidden_layer)
        #output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
        network.append(output_layer)
        return network
 
# Make a prediction with a network
def predict(network, row):
        outputs = forward_propagate(network, row)
        #print(outputs)
        indexOut=outputs.index(max(outputs))+1
        #print(indexOut)
        return indexOut
 
# Backpropagation Algorithm With Stochastic Gradient Descent
def back_propagation(train, test, l_rate, n_epoch, n_hidden):
        n_inputs = len(train[0]) - 1
        
        n_outputs = len(set([row[-1] for row in train]))
        
        network = init_net(n_inputs, n_hidden, n_outputs)
        #print("initialize network {}\n".format(network))
        neural_network_train(network, train, l_rate, n_epoch, n_outputs)
        #print("network {}\n".format(network))
        predictions = list()
        for row in test:
                prediction = predict(network, row)
                predictions.append(prediction)
        return(predictions)

        
l_rate = 0.2            
n_epoch = 100
filename = 'data.csv'
sRatio = 0.60
dataset = load_csv(filename)

for i in range(len(dataset[0])-1):
        str_column_to_float(dataset, i)
#convert class column to integers
str_column_to_int(dataset, len(dataset[0])-1)
trainingSet, testSet = splitData(dataset, sRatio)
fuzzy_threshold, fuzzy, fuzzy_test=fuzzify(trainingSet, testSet)
final_sample=process(trainingSet,fuzzy_threshold)
# for the computation of output class fuzzification

classes=[row[-1] for row in fuzzy]
Unique=np.unique(classes)
dataset_split=list()
fold_size=int(len(Unique))
for i in range(fold_size):
        fold=list()
        for row in fuzzy:
                if row[-1] == Unique[i]:
                        fold.append(row)
        dataset_split.append(fold)
i=0
mean=list()
stdev=list()
j=0
for fold in dataset_split:    
        x=list()
        y=list()
        z=list()
        x1=list()
        y1=list()
        z1=list()
        x2=list()
        y2=list()
        z2=list()
                
        for row in fold:            
                if row[-1] == Unique[j]:
                        x.append(row[0])
                        y.append(row[1])
                        z.append(row[2])
                        x1.append(row[3])
                        y1.append(row[4])
                        z1.append(row[5])
                        x2.append(row[6])
                        y2.append(row[7])
                        z2.append(row[8])
        m1=sum(x)/float(len(x))
        m2=sum(y)/float(len(y))
        m3=sum(z)/float(len(z))
        m4=sum(x1)/float(len(x1))
        m5=sum(y1)/float(len(y1))
        m6=sum(z1)/float(len(z1))
        m7=sum(x2)/float(len(x2))
        m8=sum(y2)/float(len(y2))
        m9=sum(z2)/float(len(z2))
        mean.append([m1,m2,m3,m4,m5,m6,m7,m8,m9])
        st1=sum([pow(val-m1,2) for val in x])/float(len(x)-1)
        st2=sum([pow(val-m2,2) for val in y])/float(len(y)-1)
        st3=sum([pow(val-m3,2) for val in z])/float(len(z)-1)
        st4=sum([pow(val-m4,2) for val in x1])/float(len(x1)-1)
        st5=sum([pow(val-m5,2) for val in y1])/float(len(y1)-1)
        st6=sum([pow(val-m6,2) for val in z1])/float(len(z1)-1)
        st7=sum([pow(val-m7,2) for val in x2])/float(len(x2)-1)
        st8=sum([pow(val-m8,2) for val in y2])/float(len(y2)-1)
        st9=sum([pow(val-m9,2) for val in z2])/float(len(z2)-1)
        std1=math.sqrt(st1)
        std2=math.sqrt(st2)
        std3=math.sqrt(st3)
        std4=math.sqrt(st4)
        std5=math.sqrt(st5)
        std6=math.sqrt(st6)
        std7=math.sqrt(st7)
        std8=math.sqrt(st8)
        std9=math.sqrt(st9)
        stdev.append([std1,std2,std3,std4,std5,std6,std7,std8,std9])        
        j=j+1



#print(trainingSet,len(trainingSet))
#this is fuzzify input based on class belongin granulation
s1=compute_minterm(final_sample)
#print(s1)

#The cartesian product of the minterms(POS) is computed here, the product gathers all the
#terms in SOP form. The products thus obtained are reducts.
xyz=list()
for i in itertools.product(*s1):
        #print(i)
        xyz.append(set(i))
new_xyz=list()
for i in xyz:
        if i not in new_xyz:
                new_xyz.append(i)

x={1,2,4,7}
new_set=generate_new_dataset(final_sample,list(x))
#print("\n\n{}".format(new_set))
class_one=list()
class_two=list()
class_three=list()
class_four=list()
class_five=list()
class_six=list()
hidden_layer=[]
output_layer=[]
n_hidden=minterm_for_rules(new_set,list(x))

run_algorithm(fuzzy, fuzzy_test, back_propagation, l_rate, n_epoch, n_hidden)

        
               
                        
        
        


    

