require 'torch'
require 'image'
require 'nn'
require 'cunn'
require 'cudnn'


local trainset = torch.load('cifar.torch/cifar10-train.t7')
local testset = torch.load('cifar.torch/cifar10-test.t7')

local classes = {'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}

local trainData = trainset.data:float() -- convert the data from a ByteTensor to a float Tensor.
local trainLabels = trainset.label:float():add(1)
local testData = testset.data:float()
local testLabels = testset.label:float():add(1)

-- Load and normalize data:

local redChannel = trainData[{ {}, {1}, {}, {}  }] -- this picks {all images, 1st channel, all vertical pixels, all horizontal pixels}
--print(#redChannel)

local mean = {}  -- store the mean, to normalize the test set in the future
local stdv  = {} -- store the standard-deviation for the future
for i=1,3 do -- over each image channel
    mean[i] = trainData[{ {}, {i}, {}, {}  }]:mean() -- mean estimation
    trainData[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction
    stdv[i] = trainData[{ {}, {i}, {}, {}  }]:std() -- std estimation
    trainData[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
end

-- Normalize test set using same values

for i=1,3 do -- over each image channel
    testData[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction    
    testData[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
end

do -- data augmentation module
  BatchFlip,parent = torch.class('nn.BatchFlip', 'nn.Module')
	
  function BatchFlip:__init()
    parent.__init(self)
    self.train = true
  end

  function BatchFlip:updateOutput(input)
    if self.train then
      local bs = input:size(1)
      local flip_mask = torch.randperm(bs):le(bs/2)
      for i=1,input:size(1) do
	if flip_mask[i] == 1 then image.hflip(input[i], input[i]) end
      end
    end
    self.output:set(input:cuda())
    return self.output
  end
end


---	 ### predefined constants
require 'optim'

local batchSize = 64
local optimState = {}

criterion = nn.ClassNLLCriterion():cuda()

--- ### Main evaluation

function forwardNet(data,labels)
    --another helpful function of optim is ConfusionMatrix
    local confusion = optim.ConfusionMatrix(classes)
    
    model:evaluate() -- turn of drop-out

    for i = 1, data:size(1) - batchSize, batchSize do
		local x = data:narrow(1, i, batchSize)
        local yt = labels:narrow(1, i, batchSize)
        local y = model:forward(x)
        confusion:batchAdd(y,yt)
    end
    
    confusion:updateValids()
    local avgError = 1 - confusion.totalValid
    
    return avgError
end


function testModel()
    -- Load trained net (the model)
    model = torch.load("final_trained_model") 
    w, dE_dw = model:getParameters()
    print('Number of parameters:', w:nElement())
    testError = forwardNet(testData, testLabels)
    return testError
end


testError = testModel()
print('Test error: ' .. testError)
