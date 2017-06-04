import input_data_levelDB_simulator_data_augmentation
import numpy as np
from input_data_levelDB_simulator_data_augmentation import readImageFromDB
from ast import literal_eval
def readPointFromDB(db, key, size):
  point_str = db.Get(key)
  point = literal_eval(point_str)
  return np.array([point]).T #returns shape (2, 1, 1)

def rotPoint90(point, origin, rotation):
    rotation = rotation % 4
    p_o = point - origin
    if rotation == 1:
        return np.array(-p_o[1], p_o[0]) + origin
    elif rotation == 2:
        return np.array(-p_o[0], -p_o[1]) + origin
    elif rotation == 3:
        return np.array(p_o[1], -p_o[0]) + origin
    else:
        return point

def flipHPoint(point, origin):
    return np.array(-(point - origin)[0], (point - origin)[1]) + origin

class DataSet(input_data_levelDB_simulator_data_augmentation.DataSet):
    def __init__(self, images_key, input_size, num_examples, db, validation,invert, rotate):
        super(DataSet, self).__init__(self, images_key, input_size, (), num_examples, db, validation, invert, rotate)
    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if batch_size > (self._num_examples - self._index_in_epoch):
            # Finished epoch
            print 'end epoch'
            self._epochs_completed += 1
            # Shuffle the data
            """ Shufling all the Images with a single permutation """
            random.shuffle(self._images_key)
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples

        images = np.empty((batch_size, self._input_size[0], self._input_size[1],self._input_size[2]))
        points = np.empty((batch_size, 2))
        origem = np.array(self._input_size[0] / 2, self._input_size[1] / 2)
        for n in range(batch_size):
            key=self._images_key[start+n]
            rotation=0
            inversion=0
            if self.rotate:
                rotation=key & 3
                key=int(key/4)

            if self.invert:
                inversion=key & 1
                key=int(key/2)
                
            if self._is_validation:
                images[n] = readImageFromDB(self._db,'val'+str(key),self._input_size)
                points[n] = readPointFromDB(self._db,'val'+str(key)+"point")
            else:
                images[n] = readImageFromDB(self._db,str(key),self._input_size)
                points[n] = readPointFromDB(self._db,str(key)+"point")
            
            images[n]=np.rot90(images[n], rotation)
            points[n]=rotPoint90(points[n], origem, rotation)

            if inversion:
                images[n]=np.fliplr(images[n])
                points[n]=flipHPoint(points[n], origem)
                
        return images, points #, depths #, transmission
        
class DataSetManager(input_data_levelDB_simulator_data_augmentation.DataSetManager):
    def __init__(self, config):
        self.input_size = config.input_size
        self.depth_size = config.depth_size
        self.db = leveldb.LevelDB(config.leveldb_path + 'db') 
        self.num_examples = int(self.db.Get('num_examples'))
        self.num_examples_val = int(self.db.Get('num_examples_val'))
        if config.invert:
            self.num_examples = self.num_examples * 2
            self.num_examples_val= self.num_examples_val * 2
        if config.rotate:
            self.num_examples = self.num_examples * 4
            self.num_examples_val= self.num_examples_val * 4
        self.images_key = range(self.num_examples)
        self.images_key_val = range(self.num_examples_val)
        # for i in range(self.num_examples_val):
        #     self.images_key_val[i] = 'val' + str(i)
        self.train = DataSet(self.images_key,config.input_size,config.depth_size,self.num_examples,self.db,validation=False,invert=config.invert,rotate=config.rotate)
        self.validation = DataSet(self.images_key_val,config.input_size,config.depth_size,self.num_examples_val,self.db,validation=True,invert=config.invert,rotate=config.rotate)

