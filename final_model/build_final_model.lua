--[[
Due to interest of time, please prepared the data before-hand into a 4D torch
ByteTensor of size 50000x3x32x32 (training) and 10000x3x32x32 (testing) 

mkdir t5
cd t5/
git clone https://github.com/soumith/cifar.torch.git
cd cifar.torch/
th Cifar10BinToTensor.lua

]]

require 'torch'
require 'image'
require 'nn'
require 'cunn'
require 'cudnn'


local params = {...}
local aug = params[1]
local opt = params[2]
local isFirst = params[3] or 'N'
print('**************Starting-'..aug..'--'..opt..'**************')

function saveTensorAsGrid(tensor,fileName) 
	local padding = 1
	local grid = image.toDisplayTensor(tensor/255.0, padding)
	image.save(fileName,grid)
end

local trainset = torch.load('cifar.torch/cifar10-train.t7')
local testset = torch.load('cifar.torch/cifar10-test.t7')

local classes = {'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}

local trainData = trainset.data:float() -- convert the data from a ByteTensor to a float Tensor.
local trainLabels = trainset.label:float():add(1)
local testData = testset.data:float()
local testLabels = testset.label:float():add(1)

--print(trainData:size())

--saveTensorAsGrid(trainData:narrow(1,100,36),'train_100-136.jpg') -- display the 100-136 images in dataset
--print(classes[trainLabels[100]]) -- display the 100-th image class


--  *****************************************************************
--  Let's take a look at a simple convolutional layer:
--  *****************************************************************

--[[
local img = trainData[100]:cuda()
print(img:size())

local conv = cudnn.SpatialConvolution(3, 16, 5, 5, 4, 4, 0, 0)
conv:cuda()
-- 3 input maps, 16 output maps
-- 5x5 kernels, stride 4x4, padding 0x0

print(conv)

local output = conv:forward(img)
print(output:size())
saveTensorAsGrid(output, 'convOut.jpg')

local weights = conv.weight
saveTensorAsGrid(weights, 'convWeights.jpg')
print(weights:size())
]]
--  ****************************************************************
--  Full Example - Training a ConvNet on Cifar10
--  ****************************************************************

-- Load and normalize data:

local redChannel = trainData[{ {}, {1}, {}, {}  }] -- this picks {all images, 1st channel, all vertical pixels, all horizontal pixels}
--print(#redChannel)

local mean = {}  -- store the mean, to normalize the test set in the future
local stdv  = {} -- store the standard-deviation for the future
for i=1,3 do -- over each image channel
    mean[i] = trainData[{ {}, {i}, {}, {}  }]:mean() -- mean estimation
--    print('Channel ' .. i .. ', Mean: ' .. mean[i])
    trainData[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction
    
    stdv[i] = trainData[{ {}, {i}, {}, {}  }]:std() -- std estimation
--    print('Channel ' .. i .. ', Standard Deviation: ' .. stdv[i])
    trainData[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
end

-- Normalize test set using same values

for i=1,3 do -- over each image channel
    testData[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction    
    testData[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
end


do -- data augmentation module
  if isFirst == 'Y' then 
    BatchFlip,parent = torch.class('nn.BatchFlip', 'nn.Module')
  end

	
  function BatchFlip:__init()
    parent.__init(self)
    self.train = true
  end

  function BatchFlip:updateOutput(input)
    if self.train then
      local bs = input:size(1)
      local flip_mask = torch.randperm(bs):le(bs/2)
      for i=1,input:size(1) do
		if aug == "hflip" then
			if flip_mask[i] == 1 then image.hflip(input[i], input[i]) end
		end
		if aug == "vflip" then
			if flip_mask[i] == 1 then image.vflip(input[i], input[i]) end
		end
		if aug == "crop" then
			if flip_mask[i] == 1 then cropAndScale(input[i]) end
		end
      end
    end
    self.output:set(input:cuda())
    return self.output
  end
end

function cropAndScale(destAndSrc)
  local cropped = image.crop(destAndSrc, 4, 4, 28, 28)
  return image.scale(destAndSrc, cropped)
end
--  ****************************************************************
--  Define our neural network
--  ****************************************************************




local model = nn.Sequential()


model:add(nn.BatchFlip())


local function ConvBNReLU(nInputPlane, nOutputPlane)
  model:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 3, 3, 1, 1, 1, 1))
  model:add(nn.SpatialBatchNormalization(nOutputPlane,1e-3))
  model:add(nn.ReLU(true))
  return model
end

-- ~49914 parameters
ConvBNReLU(3, 16)
ConvBNReLU(16, 16)
model:add(nn.SpatialMaxPooling(2,2,2,2):ceil())
ConvBNReLU(16, 32)
ConvBNReLU(32, 32)
model:add(nn.SpatialMaxPooling(2,2,2,2):ceil())
ConvBNReLU(32, 32)
ConvBNReLU(32, 32)
model:add(nn.SpatialMaxPooling(2,2,2,2):ceil())
ConvBNReLU(32, 32)
model:add(nn.View(32*4*4):setNumInputDims(3))  -- reshapes from a 3D tensor of 32x4x4 into 1D tensor of 32*2*2
model:add(nn.ReLU(true))
model:add(nn.Dropout(0.5))                      --Dropout layer with p=0.5
model:add(nn.Linear(32*4*4, #classes))            -- 10 is the number of outputs of the network (in this case, 10 digits)
model:add(nn.LogSoftMax())                     -- converts the output to a log-probability. Useful for classification


--[[local function ConvBNReLU(...)
  local args = {...}
  model:add(nn.SpatialConvolution(...))
  model:add(nn.SpatialBatchNormalization(args[2],1e-3))
  model:add(nn.ReLU(true))
  return model
end


-- 45966 parameters
ConvBNReLU(3, 32, 3, 3, 1, 1, 1, 1)
ConvBNReLU(32, 16, 3, 3, 1, 1, 1, 1)
ConvBNReLU(16, 32, 1, 1)
model:add(nn.SpatialMaxPooling(3,3,2,2):ceil())
model:add(nn.Dropout())
ConvBNReLU(32, 32, 3, 3, 1, 1, 1, 1)
ConvBNReLU(32, 32, 3, 3, 1, 1, 1, 1)
ConvBNReLU(32, 64, 1, 1)
model:add(nn.SpatialAveragePooling(3,3,2,2):ceil())
model:add(nn.Dropout())
ConvBNReLU(64, 32, 3, 3, 1, 1, 1, 1)
ConvBNReLU(32, 10, 1, 1)
model:add(nn.SpatialAveragePooling(8,8,1,1):ceil())


model:add(nn.View(10))  -- reshapes from a 3D tensor of 32x4x4 into 1D tensor of 32*2*2
]]

model:cuda()
criterion = nn.ClassNLLCriterion():cuda()


w, dE_dw = model:getParameters()
print('Number of parameters:', w:nElement())
print(model)

function shuffle(data,ydata) --shuffle data function
    local RandOrder = torch.randperm(data:size(1)):long()
    return data:index(1,RandOrder), ydata:index(1,RandOrder)
end

--  ****************************************************************
--  Training the network
--  ****************************************************************
require 'optim'

local batchSize = 64
local optimState = {}

function forwardNet(data,labels, train)
    --another helpful function of optim is ConfusionMatrix
    local confusion = optim.ConfusionMatrix(classes)
    local lossAcc = 0
    local numBatches = 0
    if train then
        --set network into training mode
        model:training()
    else
        model:evaluate() -- turn of drop-out
    end
    for i = 1, data:size(1) - batchSize, batchSize do
        numBatches = numBatches + 1
        --local x = data:narrow(1, i, batchSize):cuda()
        --local yt = labels:narrow(1, i, batchSize):cuda()
		local x = data:narrow(1, i, batchSize)
        local yt = labels:narrow(1, i, batchSize)
        local y = model:forward(x)
        local err = criterion:forward(y, yt)
        lossAcc = lossAcc + err
        confusion:batchAdd(y,yt)
        
        if train then
            function feval()
                model:zeroGradParameters() --zero grads
                local dE_dy = criterion:backward(y,yt)
                model:backward(x, dE_dy) -- backpropagation
            
                return err, dE_dw
            end
			if opt == "adam" then optim.adam(feval, w, optimState) end
			if opt == "sgd" then optim.sgd(feval, w, optimState) end
        end
    end
    
    confusion:updateValids()
    local avgLoss = lossAcc / numBatches
    local avgError = 1 - confusion.totalValid
    
    return avgLoss, avgError, tostring(confusion)
end

function plotError(trainError, testError, title)
	require 'gnuplot'
	local range = torch.range(1, trainError:size(1))
	gnuplot.pngfigure('testVsTrainError-'..aug..'--'..opt..'.png')
	gnuplot.plot({'trainError',trainError},{'testError',testError})
	gnuplot.xlabel('epochs')
	gnuplot.ylabel('Error')
	gnuplot.plotflush()
end

function plotLoss(trainLoss, testLoss, title)
	require 'gnuplot'
	local range = torch.range(1, trainLoss:size(1))
	gnuplot.pngfigure('testVsTrainLoss-'..aug..'--'..opt..'.png')
	gnuplot.plot({'trainLoss',trainLoss},{'testLoss',testLoss})
	gnuplot.xlabel('epochs')
	gnuplot.ylabel('Loss')
	gnuplot.plotflush()
end

---------------------------------------------------------------------

epochs = 200
trainLoss = torch.Tensor(epochs)
testLoss = torch.Tensor(epochs)
trainError = torch.Tensor(epochs)
testError = torch.Tensor(epochs)

--reset net weights
model:apply(function(l) l:reset() end)

timer = torch.Timer()


for e = 1, epochs do
    trainData, trainLabels = shuffle(trainData, trainLabels) --shuffle training data
    trainLoss[e], trainError[e] = forwardNet(trainData, trainLabels, true)
    testLoss[e], testError[e], confusion = forwardNet(testData, testLabels, false)
    
    --[[if e % 1 == 0 then
        print('Epoch ' .. e .. ':')
        print('Training error: ' .. trainError[e], 'Training Loss: ' .. trainLoss[e])
        print('Test error: ' .. testError[e], 'Test Loss: ' .. testLoss[e])
        print(confusion)
    end]]	
end

--[[local key, min_err = 1, testError[1]
for k, v in ipairs(testError) do
    if testError[k] < min_err then
        key, min_err = k, v
    end
end]]
val, i = testError:topk(1)
print(aug..'--'..opt..'--minimum test error:' .. val[1], 'get in epoch:' .. i[1])

plotError(trainError, testError, 'Classification Error')

plotLoss(trainLoss, testLoss, 'Classification Loss')

--  ****************************************************************
--  Network predictions
--  ****************************************************************


model:evaluate()   --turn off dropout

--print(classes[testLabels[10] ])
--print(testData[10]:size())
--saveTensorAsGrid(testData[10],'testImg10.jpg')
--local predicted = model:forward(testData[10]:view(1,3,32,32):cuda())
--print(predicted:exp()) -- the output of the network is Log-Probabilities. To convert them to probabilities, you have to take e^x 

-- assigned a probability to each classes
--[[for i=1,predicted:size(2) do
    print(classes[i],predicted[1][i])
end]]



--  ****************************************************************
--  Visualizing Network Weights+Activations
--  ****************************************************************

--[[
local Weights_1st_Layer = model:get(1).weight
local scaledWeights = image.scale(image.toDisplayTensor({input=Weights_1st_Layer,padding=2}),200)
saveTensorAsGrid(scaledWeights,'Weights_1st_Layer.jpg')


print('Input Image')
saveTensorAsGrid(testData[100],'testImg100.jpg')
model:forward(testData[100]:view(1,3,32,32):cuda())
for l=1,9 do
  print('Layer ' ,l, tostring(model:get(l)))
  local layer_output = model:get(l).output[1]
  saveTensorAsGrid(layer_output,'Layer'..l..'-'..tostring(model:get(l))..'.jpg')
  if ( l == 5 or l == 9 )then
	local Weights_lst_Layer = model:get(l).weight
	local scaledWeights = image.scale(image.toDisplayTensor({input=Weights_lst_Layer[1],padding=2}),200)
	saveTensorAsGrid(scaledWeights,'Weights_'..l..'st_Layer.jpg')
  end 
end
]]





print('**************End of'..aug..'--'..opt..'**************')
