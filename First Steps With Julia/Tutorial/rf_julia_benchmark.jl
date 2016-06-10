using Images, Colors, FixedPointNumbers
using DataFrames

#typeData could be either "train" or "test.
#labelsInfo should contain the IDs of each image to be read
#The images in the trainResized and testResized data files
#are 20x20 pixels, so imageSize is set to 400.
#path should be set to the location of the data files.

function read_data(typeData, labelsInfo, imageSize, path)
 #Intialize x matrix
 x = zeros(size(labelsInfo, 1), imageSize)

 for (index, idImage) in enumerate(labelsInfo[:ID])
  #Read image file
  nameFile = "$(path)/$(typeData)Resized/$(idImage).Bmp"
  img = load(nameFile)

  #Convert color images to gray images
  temp  = convert(Image{Gray}, img)

  #Transform image matrix to a vector and store
  #it in data matrix
  x[index, :] = reshape(temp, 1, imageSize)
 end
 return x
end

imageSize = 400 # 20 x 20 pixels

#Set location of data files , folders
path = "../Data"

#Read information about training data , IDs.
labelsInfoTrain = readtable("$(path)/trainLabels.csv")

#Read training matrix
xTrain = read_data("train", labelsInfoTrain, imageSize, path)
println("Finish load training data!")

#Read information about test data ( IDs ).
labelsInfoTest = readtable("$(path)/sampleSubmission.csv")

#Read test matrix
xTest = read_data("test", labelsInfoTest, imageSize, path)
println("Finish load testing data!")

#Get only first character of string (convert from string to character).
#Apply the function to each element of the column "Class"
yTrain = map(x -> x[1], labelsInfoTrain[:Class])
#Convert from character to integer
yTrain = int(yTrain)
yTrain = convert(Array, yTrain)

using DecisionTree

#Train random forest with
#20 for number of features chosen at each random split,
#50 for number of trees,
#and 1.0 for ratio of subsamp ling.
println("Training...")
model = build_forest(yTrain, xTrain, 20, 100, 0.8)

#Get predictions for test data
println("Predicting...")
predTest = apply_forest(model, xTest)

#Convert integer predictions to character
labelsInfoTest[:Class] = char(predTest)

#Save predictions
writetable("$(path)/juliaSubmission.csv", labelsInfoTest, separator=',', header=true)

#Run 4 fold cross validation
# println("Validating...")
# accuracy = nfoldCV_forest(yTrain, xTrain, 20, 50, 4, 1.0);
# println ("4 fold accuracy: $(mean(accuracy))")
