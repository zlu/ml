class ML
  @@weights = [0, 0, 0]

  def dot_product(a, b)
    [a, b].transpose.map{ |e| e = e[0]*e[1] }.inject{ |s, n| s += n }
  end

  def validation_set
    [[[1, 0, 0], 1], [[1, 0, 1], 1], [[1, 1, 0], 1], [[1, 1, 1], 0]]
  end

  def weights
    @@weights
  end

  def threshold
    0.5
  end

  def learning_rate
    0.005
  end

  def train
    w_dist = 20
    (4..10).each do |size|
      set = gen_training_set(size)
      weights = perceive_all(set)
      dist = w_distance(weights)
      w_dist = dist if w_dist > dist
      puts "final distance: " + w_dist.to_s
    end
    puts "final distance: " + w_dist.to_s
    w_dist
  end

  def perceive_all(t_set)
    puts "in per all " + t_set.size.to_s
    counter = 0
    while counter < t_set.size
      counter = 0
      t_set.each do |set|
        counter += perceive(set[0], set[1])
      end
      puts "set:counter => " + t_set.size.to_s + ":" + counter.to_s
    end
    puts "final weights: " + weights.join(":")
    weights
  end

  def perceive(input, output)
    result = dot_product(input, weights) > threshold ? 1 : 0
    diff = result - output
    return 1 if diff == 0

    input.each_with_index do |inp, index|
      weights[index] += (-1 * diff * learning_rate) if inp != 0
    end

    0
  end

  def w_distance(w)
    total_distance = 0.0
    w.each_with_index do |elem, index|
      if (index < w.length-1)
        (index+1..w.length-1).each do |n_elem|
          total_distance += (elem - w[n_elem]).abs
        end
      end
    end

    total_distance
  end

  def gen_training_set(size)
    set = [[]]
    (0..size-1).each do |i|
      temp = []
      (0..2).each do |j|
        temp[j] = rand(2)
      end
      set[i] = [temp, temp.include?(0)? 1 : 0]
    end

    p set
    set
  end
end