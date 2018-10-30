# centroid is the geometric centre

centroid(x) = x

centroid(d::Circle) = d.center
centroid(d::IntervalOrSegment) = mean((first(d),last(d)))

# dist and dist2 are the nearest distance and distance-squared between x and y

dist(x,y) = sqrt(dist2(x,y))
dist2(x,y) = dist(x,y)^2

dist2(x::Number,y::Number) = abs2(x-y)
dist2(d1::Domain,d2::Domain) = minimum(extrema2(d1,d2))

# diam is the diameter of x

diam(x) = sqrt(diam2(x))
diam2(x) = diam(x)^2

diam2(x::Number) = 1
diam(d::Circle) = 2d.radius
diam2(d::IntervalOrSegment) = abs2(first(d)-last(d))
diam2(d::UnionDomain) = maximum(extrema2(d))

# Extremal distances between domains

function Base.extrema(d1::Domain,d2::Domain)
    ext2 = extrema2(d1,d2)
    sqrt(ext2[1]),sqrt(ext2[2])
end
function extrema2(d1::Domain,d2::Domain)
    ext = extrema(d1,d2)
    ext[1]^2,ext[2]^2
end

for op in (:(Base.extrema),:extrema2)
    @eval begin
        function $op(d::UnionDomain)
            ext = [$op(d1,d2) for d1 in d,d2 in d]
            minimum(minimum(ext)),maximum(maximum(ext))
        end
        function $op(d1::UnionDomain,d2::UnionDomain)
            ext = [$op(d11,d22) for d11 in d1,d22 in d2]
            minimum(minimum(ext)),maximum(maximum(ext))
        end
        function $op(d1::Domain,d2::UnionDomain)
            ext = [$op(d1,d22) for d22 in d2]
            minimum(minimum(ext)),maximum(maximum(ext))
        end
        $op(d1::UnionDomain,d2::Domain) = $op(d2,d1)
    end
end

#
# Geometry for non-intersecting IntervalOrSegments
#
function dist2(c::Number,d::IntervalOrSegment)
    if in(c,d)
        zero(real(c))
    else
        a,b = endpoints(d)
        x1,y1 = reim(a)
        x2,y2 = reim(b)
        x3,y3 = reim(c)
        px,py = x2-x1,y2-y1
        u = ((x3-x1)px+(y3-y1)py)/(px^2+py^2)
        u = u > 1 ? 1 : u ≥ 0 ? u : 0
        dx,dy = x1+u*px-x3,y1+u*py-y3
        dx^2+dy^2
    end
end

function extrema2(d1::IntervalOrSegment,d2::IntervalOrSegment)
    a,b = d1.a,d1.b
    c,d = d2.a,d2.b
    extrema((dist2(a,d2),dist2(b,d2),dist2(c,d1),dist2(d,d1),abs2(a-c),abs2(a-d),abs2(b-c),abs2(b-d)))
end


#
# Geometry for non-intersecting Circles
#
function dist2(z::Number,d::Circle)
    if in(z,d)
        zero(real(z))
    else
        abs2(z-d.center-d.radius*cis(angle(z-d.center)))
    end
end

function extrema2(d1::Circle,d2::Circle)
    r1,r2=d1.radius,d2.radius
    c1,c2=d1.center,d2.center
    if  d1 == d2 # Circles are equal
        (0,(2d1.radius)^2)
    elseif r1>r2&&abs(c1-c2)<r1 || r1<r2&&abs(c1-c2)<r2 # Circles are interior/exterior
        extrema((abs2(c1-c2-(r1-r2)*cis(angle(c1-c2))),
                 abs2(c1-c2+(r1+r2)*cis(angle(c1-c2))),
                 abs2(c1-c2-(r1+r2)*cis(angle(c1-c2))),
                 abs2(c1-c2+(r1-r2)*cis(angle(c1-c2)))))
    else # Circles are disjoint
        extrema((abs2(c1-c2-(r1+r2)*cis(angle(c1-c2))),
                 abs2(c1-c2+(r1+r2)*cis(angle(c1-c2)))))
    end
end

#
# Geometry between non-intersecting IntervalOrSegments and Circles
#
# Consider the minimal distances from a & b to the center, adding or subtracting
# a radius in either direction. Add to these combinations the minimal distance from the center
# to the IntervalOrSegment, adding or subtracting a radius in either direction.
# The extremal distances are a subset.
#
function Base.extrema(d1::IntervalOrSegment,d2::Circle)
    a,b = d1.a,d1.b
    c,r=d2.center,d2.radius
    extrema(( dist(a,d2),dist(b,d2),dist(a,d2)+2r,dist(b,d2)+2r,dist(c,d1)+r,dist(c,d1)-r ))
end
Base.extrema(d1::Circle,d2::IntervalOrSegment) = extrema(d2,d1)
