#
# test fractals
#
using ApproxFun, SingularIntegralEquations, Base.Test

d = Interval(0.,1.)
c = Circle()

for set in (:cantor,:thincantor,:thinnercantor,:thinnestcantor)
    @eval begin
        @test $set(d,0) == d
        @test $set(d,0,4) == d
        @test $set(d,1) == Interval(0.0,1/3)∪Interval(2/3,1.0)
        @test $set(d,1,4) == Interval(0.0,1/4)∪Interval(3/4,1.0)
        @test $set(c,0) == c
        @test $set(c,1) == Arc(0.0,1.0,(-2.6179938779914944,-0.523598775598299))∪Arc(0.0,1.0,(0.5235987755982987,2.6179938779914944))
    end
end

@test cantor(d,2) == Interval(0.0,1/9)∪Interval(2/9,1/3)∪Interval(2/3,7/9)∪Interval(8/9,1.0)
@test cantor(d,2,4) == Interval(0.0,1/16)∪Interval(3/16,1/4)∪Interval(3/4,13/16)∪Interval(15/16,1.0)

@test thincantor(d,2) == Interval(0.0,1/12)∪Interval(1/4,1/3)∪Interval(2/3,3/4)∪Interval(11/12,1.0)
@test thincantor(d,2,4) == Interval(0.0,1/20)∪Interval(1/5,1/4)∪Interval(3/4,4/5)∪Interval(19/20,1.0)

@test thinnercantor(d,2) == Interval(0.0,1/27)∪Interval(8/27,1/3)∪Interval(2/3,19/27)∪Interval(26/27,1.0)
@test thinnercantor(d,2,4) == Interval(0.0,1/64)∪Interval(15/64,1/4)∪Interval(3/4,49/64)∪Interval(63/64,1.0)
@test thinnercantor(d,3,4) == Interval(0.0,1/4096)∪Interval(63/4096,1/64)∪Interval(15/64,961/4096)∪Interval(1023/4096,1/4)∪Interval(3/4,3073/4096)∪Interval(3135/4096,49/64)∪Interval(63/64,4033/4096)∪Interval(4095/4096,1/1)

@test thinnestcantor(d,2) == Interval(0.0,1/27)∪Interval(8/27,1/3)∪Interval(2/3,19/27)∪Interval(26/27,1.0)
@test thinnestcantor(d,2,4) == Interval(0.0,1/64)∪Interval(15/64,1/4)∪Interval(3/4,49/64)∪Interval(63/64,1.0)
@test thinnestcantor(d,3,4) == Interval(0.0,1/16384)∪Interval(255/16384,1/64)∪Interval(15/64,3841/16384)∪Interval(4095/16384,1/4)∪Interval(3/4,12289/16384)∪Interval(12543/16384,49/64)∪Interval(63/64,16129/16384)∪Interval(16383/16384,1/1)

@test smithvolterracantor(d,0) == d
@test smithvolterracantor(d,1) == Interval(0.0,3/8)∪Interval(5/8,1.0)
@test smithvolterracantor(d,2) == Interval(0.0,5/32)∪Interval(7/32,3/8)∪Interval(5/8,25/32)∪Interval(27/32,1.0)
