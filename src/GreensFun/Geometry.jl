
# Nearest distance from a point to a domain
dist(c::Number,d::Domain) = sqrt(dist2(c,d))
# Extremal distances between domains
function Base.extrema(d1::Domain,d2::Domain)
    ext2 = extrema2(d1,d2)
    sqrt(ext2[1]),sqrt(ext2[2])
end
function extrema2(d1::Domain,d2::Domain)
    ext = extrema(d1,d2)
    ext[1]^2,ext[2]^2
end

#
# Geometry for non-intersecting Intervals
#
function dist2(c::Number,d::Interval)
    if in(c,d)
        zero(real(c))
    else
        a,b = d.a,d.b
        x1,y1 = real(a),imag(a)
        x2,y2 = real(b),imag(b)
        x3,y3 = real(c),imag(c)
        px,py = x2-x1,y2-y1
        u = ((x3-x1)px+(y3-y1)py)/(px^2+py^2)
        u = u > 1 ? 1 : u ≥ 0 ? u : 0
        dx,dy = x1+u*px-x3,y1+u*py-y3
        dx^2+dy^2
    end
end

function extrema2(d1::Interval,d2::Interval)
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
# Geometry between non-intersecting Intervals and Circles
#
# Consider the minimal distances from a & b to the center, adding or subtracting
# a radius in either direction. Add to these combinations the minimal distance from the center
# to the Interval, adding or subtracting a radius in either direction.
# The extremal distances are a subset.
#
function Base.extrema(d1::Interval,d2::Circle)
    a,b = d1.a,d1.b
    c,r=d2.center,d2.radius
    extrema(( dist(a,d2),dist(b,d2),dist(a,d2)+2r,dist(b,d2)+2r,dist(c,d1)+r,dist(c,d1)-r ))
end
Base.extrema(d1::Circle,d2::Interval) = extrema(d2,d1)
