import tensorflow as tf 

# This function keeps the initial learning rate for the first ten epochs.
# The learning rate decreases exponentially after the first ten epochs.
def scheduler(epoch, lr):
  if epoch < 10:
    return lr
  else:
    return lr * tf.math.exp(-0.1)

def return_callbacks(model_file, metric):
    callbacks = [
        # Save the best model (the one with the lowest value for the specified metric).
        tf.keras.callbacks.ModelCheckpoint(
            model_file, save_best_only=True, monitor=metric, mode='min'
        ), 
        tf.keras.callbacks.LearningRateScheduler(scheduler)
    ]
    return callbacks