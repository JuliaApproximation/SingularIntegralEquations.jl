using ApproxFun, SingularIntegralEquations, Test

@testset "Fractal" begin
    d = Segment(0.,1.)
    c = Circle()

    for set in (:cantor,:thincantor,:thinnercantor,:thinnestcantor)
        @eval begin
            d = Segment(0.,1.)
            c = Circle()
            @test $set(d,0) == d
            @test $set(d,0,4) == d
            @test $set(d,1) == Segment(0.0,1/3) ∪ Segment(2/3,1.0)
            @test $set(d,1,4) == Segment(0.0,1/4) ∪ Segment(3/4,1.0)
            @test $set(c,0) == c
            @test $set(c,1) == Arc(0.0,1.0,(-2.6179938779914944,-0.523598775598299)) ∪
                                Arc(0.0,1.0,(0.5235987755982987,2.6179938779914944))
        end
    end

    @test cantor(d,2) == Segment(0.0,1/9)∪Segment(2/9,1/3)∪Segment(2/3,7/9)∪Segment(8/9,1.0)
    @test cantor(d,2,4) == Segment(0.0,1/16)∪Segment(3/16,1/4)∪Segment(3/4,13/16)∪Segment(15/16,1.0)

    @test thincantor(d,2) == Segment(0.0,1/12)∪Segment(1/4,1/3)∪Segment(2/3,3/4)∪Segment(11/12,1.0)
    @test thincantor(d,2,4) == Segment(0.0,1/20)∪Segment(1/5,1/4)∪Segment(3/4,4/5)∪Segment(19/20,1.0)

    @test thinnercantor(d,2) == Segment(0.0,1/27)∪Segment(8/27,1/3)∪Segment(2/3,19/27)∪Segment(26/27,1.0)
    @test thinnercantor(d,2,4) == Segment(0.0,1/64)∪Segment(15/64,1/4)∪Segment(3/4,49/64)∪Segment(63/64,1.0)
    @test thinnercantor(d,3,4) == Segment(0.0,1/4096)∪Segment(63/4096,1/64)∪Segment(15/64,961/4096)∪Segment(1023/4096,1/4)∪Segment(3/4,3073/4096)∪Segment(3135/4096,49/64)∪Segment(63/64,4033/4096)∪Segment(4095/4096,1/1)

    @test thinnestcantor(d,2) == Segment(0.0,1/27)∪Segment(8/27,1/3)∪Segment(2/3,19/27)∪Segment(26/27,1.0)
    @test thinnestcantor(d,2,4) == Segment(0.0,1/64)∪Segment(15/64,1/4)∪Segment(3/4,49/64)∪Segment(63/64,1.0)
    @test thinnestcantor(d,3,4) == Segment(0.0,1/16384)∪Segment(255/16384,1/64)∪Segment(15/64,3841/16384)∪Segment(4095/16384,1/4)∪Segment(3/4,12289/16384)∪Segment(12543/16384,49/64)∪Segment(63/64,16129/16384)∪Segment(16383/16384,1/1)

    @test smithvolterracantor(d,0) == d
    @test smithvolterracantor(d,1) == Segment(0.0,3/8)∪Segment(5/8,1.0)
    @test smithvolterracantor(d,2) == Segment(0.0,5/32)∪Segment(7/32,3/8)∪Segment(5/8,25/32)∪Segment(27/32,1.0)
end
