MAX_DISTANCE = 50

class Person:
    def __init__(self, id, centroid, bounding_box,  cur_loss_count = 0):
        self.id = id 
        self.centroid = centroid 
        self.cur_loss_count = cur_loss_count # Num of frames person hasn't been seen for
        self.paths = []
        self.bounding_box = bounding_box

    def is_lost(self, max_loss_count):
        if (self.cur_loss_count > max_loss_count):
            return True
        return False

    def __str__(self):
        return f"ID: {self.id} Centroid: {self.centroid} Cur_loss: {self.cur_loss_count} Paths: {self.paths} Box: {self.bounding_box}"
    
class Tracker:
    def __init__(self, maxLost=30):
        """
        Initialize the tracker.

        Parameters:
        maxLost (int): Maximum number of frames an object can be lost before deregistration.
        """
        self.max_loss = maxLost
        self.cur_id = 0
        # [id, centroid, cur_loss_count]
        self.trackers = [] # Stores Person() Objects 

    def __str__(self):
        print_str = ""
        for person in self.trackers:
            print_str += str(person) + '\n'
        return print_str

    def register(self, centroid, bounding_box):
        """
        Register a new object with the next available ID.

        Parameters:
        centroid: The centroid of the new object to register (x, y).

        Returns:
        None
        """

        # Check if ID already exists/has existed 
        person = Person(self.cur_id, centroid, bounding_box)
        
        self.trackers.append(person)
        self.cur_id += 1


    def deregister(self, objectID):
        """
        Deregister an object, removing it from tracking.

        Parameters:
        objectID: The ID of the object to deregister.

        Returns:
        None
        """
        for i, person in enumerate(self.trackers):
            if person.id == objectID:
                del self.trackers[i]


    def update(self, inputCentroids, filtered_bounding_boxes):
        """
        Update the tracked objects with new centroid information from the current frame.

        Parameters:
        inputCentroids: list of centroids detected in the current frame.

        Returns:
        Updated objects with their current centroid positions.
        """
            
        for box, centroid in enumerate(inputCentroids):
            min_distance = float('inf')
            index_temp = -1
            for i, person in enumerate(self.trackers): # Find the person closest to the curr centroid
                distance = ((person.centroid[0] - centroid[0])**2 + (person.centroid[1] - centroid[1])**2)**.5
                if (distance < MAX_DISTANCE) and (distance < min_distance):
                    index_temp = i # Keep track of closest person's index in trackers to curr centroid
                    min_distance = distance # Find person closest to centroid
                    
            if index_temp != -1: # If you found a person within max distance to the centroid
                self.trackers[index_temp].paths.append(centroid) # Store the previous centroid
                self.trackers[index_temp].centroid = centroid # Assign new centroid to current person
                self.trackers[index_temp].bounding_box = filtered_bounding_boxes[box]
                self.trackers[index_temp].cur_loss_count = 0
            else:
                self.register(centroid, filtered_bounding_boxes[box])

        # Deregister person if lost for more than max_loss
        for i, person in enumerate(self.trackers):
            person.cur_loss_count += 1
            if person.cur_loss_count > self.max_loss:
                del self.trackers[i]
                #self.deregister(person.id) Kartik Method


        return self.trackers
