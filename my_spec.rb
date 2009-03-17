require "spec"
require "ml"

describe "ML#do_product" do
  it "should produce zero" do
    ML.new.dot_product([0], [1]).should == 0
  end

  it "should produce 4321" do
    ML.new.dot_product([1,2,3,4],[1,10,100,1000]).should == 4321
  end

  it "should produce 1 with input set 1" do
    ml = ML.new
    ml.perceive([1, 0, 0], 1).should == 1
  end

  it "should produce 1 with input set 2" do
    ML.new.perceive([1,0,0], 1).should == 1
  end

  it "should produce 1 with input set 3" do
    ML.new.perceive([1,1,0], 1).should == 1
  end

  it "should produce 0 with input set 4" do
    ML.new.perceive([1,1,1], 0).should == 1
  end

  it "should produce correct results for all input sets" do
    ML.new.perceive_all(ML.new.validation_set)
  end

  it "should return 1 for the distance amongst weights of verification set" do
    ML.new.w_distance([0.8, -0.2, -0.1]).should == 2
  end

  it "should generate random training set" do
    r_training_set = ML.new.gen_training_set(4)
    r_training_set.should_not be_nil
    r_training_set.size.should == 4
  end

  it "should perceive with training sets and return the smallest w_distance" do
    ML.new.train.should_not be_nil
  end
end
