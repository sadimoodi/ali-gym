import os, imagesize, json, ray, time

ray.init()

@ray.remote
class imagesizes:
    def __init__(self):
        self.images = {}
        self.get_images_sizes()

    def get_images_sizes(self):
        for filename in os.scandir('C://Users//ali.khankan//Downloads//notebooks//Detectron//mydataset//train'):
            width, height = imagesize.get(filename.path)
            self.images[filename.name] = {'height': height, 'width': width}

    def get_images(self):
        return self.images

tic = time.time()
myimages = imagesizes.remote()
results = ray.get(myimages.get_images.remote())

with open('C://Users//ali.khankan//Downloads//notebooks//Detectron//mydataset//train//_annotations_train.json', 'r') as f:
    data = json.load(f)

for image in data['images']:
    file_name = image['file_name']
    json_height = image['height']
    json_width = image['width']
    height, width = results[file_name]['height'], results[file_name]['width']
    
    if json_width != width:
        print (f'image: {file_name} has a wrong json width: {json_width}, real width: {width}')
        image['width'] = width
    if json_height !=height:
        print (f'image: {file_name} has a wrong json height: {json_height}, real height: {height}')
        image['height'] = height
    
print ('time taken: {}'.format(time.time() - tic))

with open('_annotations_train.json', 'w') as outfile:
    json.dump(data, outfile)
print ('Finished, writing new JSON file.')
