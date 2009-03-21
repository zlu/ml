class ML
  @@weights_for_binary = [0, 0, 0]
  @@weights_for_linear = [0.0]

  def weights_for_binary
    @@weights_for_binary
  end

  def weights_for_linear
    @@weights_for_linear
  end

  def binary_training_set
    [[[1, 0, 0], 1], [[1, 0, 1], 1], [[1, 1, 0], 1], [[1, 1, 1], 0]]
  end

  def linear_training_set
    [[[1.0], 2.0], [[1.5], 3.0], [[2.0], 4.0]]
  end

  def threshold
    0.5
  end

  def dot_product(input, weights)
    [input, weights].transpose.map{ |e| e = e[0] * e[1] }.inject{ |s, n| s += n }
  end

  def binary_train_all
    train_all(binary_training_set, weights_for_binary, lambda{ |a,b| dot_product(a, b) > threshold ? 1 : 0 })
  end

  def linear_train_all
    train_all(linear_training_set, weights_for_linear, lambda{ |a,b| dot_product(a, b) })
  end

  def sigmoid_train_all

  end

  def stop_condition(training_set, counter, repetition, weights)
    counter == training_set.size  || ( repetition > 1500 && rms(weights) < 0.01)
  end

  def rms(w)
    Math.sqrt(w.inject(0){ |s, n| s += n * n } / w.size)
  end

  def train_all(training_set, weights, activation_function)
    counter = 0
    repetition = 0
    learning_rate = 0.9
    while !stop_condition(training_set, counter, repetition, weights)
      counter = 0
      repetition += 1
      training_set.each do |set|
        counter += train(set[0], set[1], learning_rate, weights, activation_function)
      end
      learning_rate *= 0.99
      p learning_rate
      puts "intermediary weights: " + weights.join(":")
    end
    puts "final weights: " + weights.join(":")
  end

  def train(input, expected_output, learning_rate, weights, activation_function)
    actual_output = activation_function.call(input, weights)
    train_step = (expected_output - actual_output) * learning_rate
    return 1 if train_step == 0

    input.each_with_index do |inp, index|
      weights[index] += inp * train_step
    end

    0
  end
end