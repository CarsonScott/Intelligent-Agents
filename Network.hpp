#ifndef NETWORK_HPP_INCLUDED
#define NETWORK_HPP_INCLUDED

#define Array std::vector<float>
#define Matrix std::vector<std::vector<float>>

Array makeArray(int size, float v=0){
    Array a;
    for(int i = 0; i < size; i++){
        a.push_back(v);
    }
    return a;
}

Matrix makeMatrix(int rows, int cols, float v=0){
    Matrix matrix;
    for(int i = 0; i < rows; i++){
        Array a = makeArray(cols, v);
        matrix.push_back(a);
    }
    return matrix;
}

Array combine(Array a, Array b){
    Array array;
    for(int i = 0; i < a.size(); i++){
        array.push_back(a[i]);
    }
    for(int i = 0; i < b.size(); i++){
        array.push_back(b[i]);
    }
    return array;
}

class Network{
    Array inputs;
    Array outputs;
    Matrix trainingInputs;
    Matrix trainingOutputs;
    Array errors;
    Matrix ihWeights;
    Array ihBiases;
    Matrix hoWeights;
    Array hoBiases;
    Array ihSums;
    Array hoSums;
    Array ihOutputs;
    Array oGrads;
    Array hGrads;
    Matrix ihPrevWeightsDelta;
    Array ihPrevBiasesDelta;
    Matrix hoPrevWeightsDelta;
    Array hoPrevBiasesDelta;

    int numInput;
    int numHidden;
    int numOutput;
public:
    void create(int nInput, int nHidden, int nOutput){
        numInput = nInput;
        numHidden = nHidden;
        numOutput = nOutput;

        inputs = makeArray(numInput);
        ihWeights = makeMatrix(numInput, numHidden);
        ihBiases = makeArray(numHidden);
        ihSums = makeArray(numHidden);
        ihOutputs = makeArray(numHidden);

        outputs = makeArray(numOutput);
        hoWeights = makeMatrix(numHidden, numOutput);
        hoBiases = makeArray(numOutput);
        hoSums = makeArray(numOutput);

        oGrads = makeArray(numOutput);
        hGrads = makeArray(numHidden);

        ihPrevWeightsDelta = makeMatrix(numInput, numHidden);
        ihPrevBiasesDelta = makeArray(numHidden);

        hoPrevWeightsDelta = makeMatrix(numHidden, numOutput);
        hoPrevBiasesDelta = makeArray(numOutput);

        for(int i = 0; i < numHidden; i++){
            for(int j = 0; j < numInput; j++){
                float num = random(10, 1);
                ihWeights[j][i] = num/10;
            }
            float num = random(10, 1);
            ihBiases[i] = num/10;
        }

        for(int i = 0; i < numOutput; i++){
            for(int j = 0; j < numHidden; j++){
                float num = random(10, 1);
                hoWeights[j][i] = num/10;
            }
            float num = random(10, 1);
            hoBiases[i] = num/10;
        }
    }

    float activation(float x){
        if(x > 0)
            return 1 / (1 + exp(-x));
        else
            return 0;
    }

    float activationDerivative(float out){
        return out * (1 - out);
    }

    Array computeErrors(Array targetOutputs){
        Array e;
        for(int i = 0; i < targetOutputs.size(); i++){
            float o = outputs[i];
            float t = targetOutputs[i];
            e.push_back(pow(t-o, 2));
        }
        return e;
    }

    float getError(){
        float error = 0;
        for(int i = 0; i < errors.size(); i++){
            error += errors[i];
        }
        return error;
    }

    Array getWeights(){
        Array weights;
        for(int i = 0; i < numInput; i++){
            for(int j = 0; j < numHidden; j++){
                weights.push_back(ihWeights[i][j]);
                weights.push_back(ihBiases[j]);
            }
        }

        for(int i = 0; i < numHidden; i++){
            for(int j = 0; j < numOutput; j++){
                weights.push_back(hoWeights[i][j]);
                weights.push_back(hoBiases[i]);
            }
        }
        return weights;
    }

    void setWeights(Array weights){
        int counter = 0;
        for(int i = 0; i < numInput; i++){
            for(int j = 0; j < numHidden; j++){
                ihWeights[i][j] = weights[counter];
                counter ++;

                ihBiases[j] = weights[counter];
                counter ++;
            }
        }

        for(int i = 0; i < numHidden; i++){
            for(int j = 0; j < numOutput; j++){
                hoWeights[i][j] = weights[counter];
                counter ++;

                hoBiases[j] = weights[counter];
                counter ++;
            }
        }
    }

    Array computeOutputs(Array xValues){
        for(int i = 0; i < numHidden; i++){
            float sum = 0;
            for(int j = 0; j < numInput; j++){
                float in = xValues[j];
                float w = ihWeights[j][i];
                sum += in*w;
            }
            sum += ihBiases[i];
            float out = activation(sum);
            ihOutputs[i] = out;
            ihSums[i] = sum;
        }

        Array yValues;
        for(int i = 0; i < numOutput; i++){
            float sum = 0;
            for(int j = 0; j < numHidden; j++){
                float in = ihOutputs[j];
                float w = hoWeights[j][i];
                sum += in*w;
            }
            sum += hoBiases[i];
            float out = activation(sum);
            yValues.push_back(out);
            hoSums[i] = sum;
        }
        return yValues;
    }

    Array computeOutputGradients(Array targetOutputs){
        Array gradients;
        for(int i = 0; i < numOutput; i++){
            float t = targetOutputs[i];
            float o = outputs[i];
            float g = activationDerivative(o) * (t-o);
            gradients.push_back(g);
        }
        return gradients;
    }

    Array computeHiddenGradients(){
        Array gradients;
        for(int i = 0; i < numHidden; i++){
            float o = ihOutputs[i];
            float g = activationDerivative(o);
            for(int j = 0; j < numOutput; j++){
                float og = oGrads[j];
                float w = hoWeights[i][j];
                g += og*w;
            }
            gradients.push_back(g);
        }
        return gradients;
    }

    Array computeHiddenDeltas(float eta, float a){
        Array deltas;
        for(int i = 0; i < numInput; i++){
            for(int j = 0; j < numHidden; j++){
                float g = hGrads[j];
                float in = inputs[i];

                float d = eta*g*in;
                float m = ihPrevWeightsDelta[i][j]*a;
                deltas.push_back(m+d);
                ihPrevWeightsDelta[i][j] = d;

                d = eta*g;
                m = ihPrevBiasesDelta[j]*a;
                deltas.push_back(m+d);
                ihPrevBiasesDelta[j] = d;
            }
        }
        return deltas;
    }

    Array computeOutputDeltas(float eta, float a){
        Array deltas;
        for(int i = 0; i < numHidden; i++){
            for(int j = 0; j < numOutput; j++){
                float g = oGrads[j];
                float in = ihOutputs[i];

                float d = eta*g*in;
                float m = hoPrevWeightsDelta[i][j]*a;

                deltas.push_back(m+d);
                hoPrevWeightsDelta[i][j] = d;

                d = eta*g;
                m = hoPrevBiasesDelta[j];
                deltas.push_back(m+d);
                hoPrevBiasesDelta[j] = d;
            }
        }
        return deltas;
    }

    void updateWeights(Array tValues, float eta, float a){
        errors = computeErrors(tValues);
        oGrads = computeOutputGradients(tValues);
        hGrads = computeHiddenGradients();

        Array hiddenDeltas = computeHiddenDeltas(eta, a);
        Array outputDeltas = computeOutputDeltas(eta, a);

        Array deltas = combine(hiddenDeltas, outputDeltas);
        Array weights = getWeights();

        for(int i = 0; i < weights.size(); i++){
            weights[i] += deltas[i];
        }
        setWeights(weights);
    }

    void train(Matrix sampleInputs, Matrix targetOutputs){
        for(int i = 0; i < sampleInputs.size(); i++){
            inputs = sampleInputs[i];
            outputs = computeOutputs(sampleInputs[i]);
            updateWeights(targetOutputs[i], 0.5, 0.3);
        }
    }

    void update(Array in, int reward){
        if(reward > 0 && random(5) % 2 == 0){
            trainingInputs.push_back(inputs);
            trainingOutputs.push_back(outputs);
            updateWeights(outputs, 0.5, 0.3);
        }

        if(trainingInputs.size() > 300){
            trainingInputs.erase(trainingInputs.begin());
            trainingOutputs.erase(trainingOutputs.begin());
        }

        inputs = in;
        outputs = computeOutputs(inputs);
    }

    Array getOutputs(){
        return outputs;
    }

    void learn(){
        train(trainingInputs, trainingOutputs);
    }
};

#endif // NETWORK_HPP_INCLUDED
