/*****************************************************************************/  
template<class T, class S>  
double Ransac<T,S>::compute(std::vector<S> &parameters,   
                                                      ParameterEsitmator<T,S> *paramEstimator ,   
                                                    std::vector<T> &data,   
                                                    int numForEstimate)  
{  
    std::vector<T *> leastSquaresEstimateData;  
    int numDataObjects = data.size();  
    int numVotesForBest = -1;  
    int *arr = new int[numForEstimate];// numForEstimate表示拟合模型所需要的最少点数，对本例的直线来说，该值为2  
    short *curVotes = new short[numDataObjects];  //one if data[i] agrees with the current model, otherwise zero  
    short *bestVotes = new short[numDataObjects];  //one if data[i] agrees with the best model, otherwise zero  
      
  
              //there are less data objects than the minimum required for an exact fit  
    if(numDataObjects < numForEstimate)   
        return 0;  
        // 计算所有可能的直线，寻找其中误差最小的解。对于100点的直线拟合来说，大约需要100*99*0.5=4950次运算，复杂度无疑是庞大的。一般采用随机选取子集的方式。  
    computeAllChoices(paramEstimator,data,numForEstimate,  
                                        bestVotes, curVotes, numVotesForBest, 0, data.size(), numForEstimate, 0, arr);  
  
       //compute the least squares estimate using the largest sub set  
    for(int j=0; j<numDataObjects; j++) {  
        if(bestVotes[j])  
            leastSquaresEstimateData.push_back(&(data[j]));  
    }  
        // 对局内点再次用最小二乘法拟合出模型  
    paramEstimator->leastSquaresEstimate(leastSquaresEstimateData,parameters);  
  
    delete [] arr;  
    delete [] bestVotes;  
    delete [] curVotes;   
  
    return (double)leastSquaresEstimateData.size()/(double)numDataObjects;  
} 
