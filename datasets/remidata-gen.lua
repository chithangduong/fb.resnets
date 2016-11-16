--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Script to compute list of ImageNet filenames and classes
--
--  This generates a file gen/imagenet.t7 which contains the list of all
--  ImageNet training and validation images and their classes. This script also
--  works for other datasets arragned with the same layout.
--

local sys = require 'sys'
local ffi = require 'ffi'
local tds = require 'tds'

local M = {}

--function M.findClasses(dir)
local function findClasses(dir)
   local dirs = paths.dir(dir)
   table.sort(dirs)

   local classList = {}
   local classToIdx = {}
   for _ ,class in ipairs(dirs) do
        if(paths.dirp(paths.concat(dir, class))) then
            if not classToIdx[class] and class ~= '.' and class ~= '..' then
                table.insert(classList, class)
                classToIdx[class] = #classList
            end
        end
   end

   -- assert(#classList == 1000, 'expected 1000 ImageNet classes')
   return classList, classToIdx
end

--function M.findImages(dir, classToIdx)
local function findImages(dir, classToIdx)
   local imagePath = torch.CharTensor()
   local imageClass = torch.LongTensor()

   ----------------------------------------------------------------------
   -- Options for the GNU and BSD find command
   local extensionList = {'jpg', 'png', 'jpeg', 'JPG', 'PNG', 'JPEG', 'ppm', 'PPM', 'bmp', 'BMP'}
   local findOptions = ' -iname "*.' .. extensionList[1] .. '"'
   for i=2,#extensionList do
      findOptions = findOptions .. ' -o -iname "*.' .. extensionList[i] .. '"'
   end

   -- Find all the images using the find command
   local f = io.popen('find -L ' .. dir .. findOptions)

   -- print("Class list")
   -- print(classToIdx)
   local maxLength = -1
   local img, cat = tds.Hash(), tds.Vec()

   for class in paths.iterdirs(dir) do
      cat:insert(class)
      local imgpath = tds.Hash()
      local dirname = dir .. '/' .. class
      for image in paths.files(dirname, 'jpg') do
        imgpath[#imgpath+1] = dirname .. '/' .. image
        maxLength = math.max(maxLength, #imgpath[#imgpath] + 1)
	    -- print(imgpath[#imgpath])
      end
      img[#cat] = imgpath
    end

   -- print("Number of files " .. count)
   f:close()

   -- Split to training/validation sets
	--print(torch.initialSeed())
   local valid_rate = 0.1
   local valImagePaths, valImageClasses = {}, {}
   local trainImagePaths, trainImageClasses = {}, {}
   local gen = torch.Generator()
   torch.manualSeed(gen,torch.initialSeed())
   for cId=1,#img do
        local size = #img[cId]
        -- print("Size " .. size)
        local imgCat = img[cId]
        local nvalid = math.floor( size*valid_rate )
        -- print("Nvalid " .. nvalid)
        local ntrain = size-nvalid
        -- print("Ntrain " ..ntrain)
        local rand = torch.randperm(gen,size)
	    --print(rand)
        local validperm = rand:narrow(1,1,nvalid)
        local trainperm = rand:narrow(1,1+nvalid,ntrain)
        -- print(validperm)
        -- print(trainperm)
        for i=1,nvalid do
            p = imgCat[validperm[i]]
            table.insert(valImagePaths,p)
            table.insert(valImageClasses, i)
        end
        for i=1,ntrain do
            p = imgCat[trainperm[i]]
            table.insert(trainImagePaths, p)
            table.insert(trainImageClasses, i)
        end
   end

   -- Convert the generated list to a tensor for faster loading
--    local nImages = #imagePaths
--    local imagePath = torch.CharTensor(nImages, maxLength):zero()
--    for i, path in ipairs(imagePaths) do
--       ffi.copy(imagePath[i]:data(), path)
--    end

--    local imageClass = torch.LongTensor(imageClasses)
    local trainImagePath = torch.CharTensor(#trainImagePaths, maxLength):zero()
    for i, path in ipairs(trainImagePaths) do
        ffi.copy(trainImagePath[i]:data(), path)
    end
    local trainImageClass = torch.LongTensor(trainImageClasses)

    local valImagePath = torch.CharTensor(#valImagePaths, maxLength):zero()
    -- print("Validation image paths")
    for i, path in ipairs(valImagePaths) do
        ffi.copy(valImagePath[i]:data(), path)
        -- print(path)
    end
    local valImageClass = torch.LongTensor(valImageClasses)

   return trainImagePath, trainImageClass, valImagePath, valImageClass
end

function M.exec(opt, cacheFile)
   -- find the image path names
   local imagePath = torch.CharTensor()  -- path to each image in dataset
   local imageClass = torch.LongTensor() -- class index of each image (class index in self.classes)

   local trainDir = paths.concat(opt.data, 'train')
   -- local valDir = paths.concat(opt.data, 'val')
   assert(paths.dirp(trainDir), 'train directory not found: ' .. trainDir)
   -- assert(paths.dirp(valDir), 'val directory not found: ' .. valDir)

   print("=> Generating list of images")
   local classList, classToIdx = findClasses(trainDir)

   -- print(" | finding all validation images")
   -- local valImagePath, valImageClass = findImages(valDir, classToIdx)

   print(" | finding all training images")
   local trainImagePath, trainImageClass, valImagePath, valImageClass = findImages(trainDir, classToIdx)

   local info = {
      basedir = opt.data,
      classList = classList,
      train = {
         imagePath = trainImagePath,
         imageClass = trainImageClass,
      },
      val = {
         imagePath = valImagePath,
         imageClass = valImageClass,
      },
   }

   print(" | saving list of images to " .. cacheFile)
   torch.save(cacheFile, info)
   return info
end

return M
