#include "model.h"

double MSE(arma::cube& pred, arma::cube& Y)
{
  return metric::SquaredEuclideanDistance::Evaluate(pred, Y) / (Y.n_elem);
}

template<typename InputDataType = arma::mat,
         typename DataType = arma::cube,
         typename LabelType = arma::cube>
void CreateTimeSeriesData(InputDataType dataset,
                          DataType& X,
                          LabelType& y,
                          const size_t rho)
{
  for (size_t i = 0; i < dataset.n_cols - rho; i++)
  {
    X.subcube(arma::span(), arma::span(i), arma::span()) =
        dataset.submat(arma::span(), arma::span(i, i + rho - 1));
    y.subcube(arma::span(), arma::span(i), arma::span()) =
        dataset.submat(arma::span(dataset.n_rows-1), arma::span(i, i + rho - 1));
  }
}

int main(int argc, char **argv)
{
    char *end;
    string temp;
    int option_index = 0;
    int opt;
    arma::mat dataset, testset;
    arma::cube trainX, trainY, testX, testY, predY;

    while ((opt = getopt_long (argc, argv, "b:d:e:h:m:o:r:s:t:", long_options, &option_index)) != EOF)
    {
        switch(opt)
        {
            case 'b': 
                batchSize = strtoul(optarg, &end, 10);
                break;
            case 'd': 
                dataFile.append(optarg);
                train = true;
                break;
            case 'e':
                epoch = strtoul(optarg, &end, 10);
                break;
            case 'h': 
                hidden = strtoul(optarg, &end, 10);
                break;
            case 'm': 
                modelFile.append(optarg);
                load = true;
                break;
            case 'o': 
                outputFile.append(optarg);
                break;
            case 'p': 
                predFile.append(optarg);
                output = true;
                break;
            case 'r': 
                rho = strtoul(optarg, &end, 10);
                break;
            case 's': 
                temp.append(optarg);
                stepSize = stod(temp);
                break;
            case 't': 
                testFile.append(optarg);
                test = true
                break;
            default: 
                perror("Error parsing Flags\n");
                LOG("Error parsing Flags\n");
                exit(1);
        }
    }
    LOG("Parsed Flags\n");

    LOG("#################Model Attributes#################\n");
    LOG("Batch Size: "<<batchSize<<"\n");
    LOG("Epoch: "<<epoch<<"\n");
    LOG("Hidden: "<<hidden<<"\n");
    LOG("Rho: "<<rho<<"\n");
    LOG("Step Size: "<<stepSize<<"\n");
    LOG("Data File: "<<dataFile<<"\n");
    LOG("Test File: "<<testFile<<"\n");
    LOG("Model File: "<<modelFile<<"\n");
    LOG("Prediction File: "<<predFile<<"\n");
    LOG("Output File: "<<outputFile<<"\n");

    size_t inputSize = 5, outputSize = 1;
    RNN<CrossEntropyError<>, HeInitialization> model(rho);


    if(load)
    {
        data::Load(modelFile, "LSTMMulti", model);
        LOG("Loaded Model from "<<modelFile<<"\n");
    }
    else 
    {
        model.Add<IdentityLayer<>>();
        model.Add<LSTM<>>(inputSize, hidden, rho);
        model.Add<Dropout<>>(0.5);
        model.Add<LeakyReLU<>>();
        model.Add<LSTM<>>(hidden, hidden, rho);
        model.Add<Dropout<>>(0.5);
        model.Add<LeakyReLU<>>();
        model.Add<LSTM<>>(hidden, hidden, rho);
        model.Add<LeakyReLU<>>();
        model.Add<Linear<>>(hidden, outputSize);
        LOG("Defined new RNN Model\n");
    }

    if(train)
    {
        data::Load(dataFile, dataset, true);
        LOG("Training Begins\n");
        trainX.set_size(inputSize, trainData.n_cols - rho + 1, rho);
        trainY.set_size(outputSize, trainData.n_cols - rho + 1, rho);
        CreateTimeSeriesData(dataset, trainX, trainY, rho);

        ens::Adam optimizer(
            stepSize,  // Step size of the optimizer.
            batchSize, // Batch size. Number of data points that are used in each
                        // iteration.
            0.9,        // Exponential decay rate for the first moment estimates.
            0.999,      // Exponential decay rate for the weighted infinity norm
                        // estimates.
            1e-8, // Value used to initialise the mean squared gradient parameter.
            trainData.n_cols * epoch, // Max number of iterations.
            1e-8,                      // Tolerance.
            true);

        optimizer.Tolerance() = -1;

        model.Train(trainX,
                trainY,
                optimizer,
                // PrintLoss Callback prints loss for each epoch.
                ens::PrintLoss(),
                // Progressbar Callback prints progress bar for each epoch.
                ens::ProgressBar(),
                // Stops the optimization process if the loss stops decreasing
                // or no improvement has been made. This will terminate the
                // optimization once we obtain a minima on training set.
                ens::EarlyStopAtMinLoss());

        LOG("Training Ends\n");

        data::Save(outputFile, "LSTMMulti", model);
        LOG("Saved Model to "<<outputFile<<"\n");
    }

    if(test && (train || load))
    {
        LOG("Testing Begins\n");
        data::Load(testFile, testset, true);
        testX.set_size(inputSize, testData.n_cols - rho + 1, rho);
        testY.set_size(outputSize, testData.n_cols - rho + 1, rho);
        CreateTimeSeriesData(testset, testX, testY, rho);

        model.Predict(testX, predY);
        double predMSE = MSE(predY, testY);
        LOG("Mean Squared Error on Prediction data points: "<<predMSE<<"\n");

        if(output)
        {
            LOG("Saving Predictions to "<<predFile<<"\n");
            arma::mat flatDataAndPreds = testX.slice(testX.n_slices - 1);
            flatDataAndPreds.rows(flatDataAndPreds.n_rows - 1) = predictions.slice(predictions.n_slices - 1);    
            data::Save(predFile, flatDataAndPreds);
        }
    }

    return 0;
}