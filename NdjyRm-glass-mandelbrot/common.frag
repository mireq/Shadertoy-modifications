const float pi = radians(180.0);
const float twoPi = radians(360.0);

mat2 rotate(float theta)
{
    vec2 cs = vec2(cos(theta), sin(theta));
    return mat2(cs.x, cs.y, -cs.y, cs.x);
}

mat3 xyRotate(float theta)
{
    vec2 cs = vec2(cos(theta), sin(theta));
    return mat3(
        cs.x, cs.y, 0,
        -cs.y, cs.x, 0,
        0, 0, 1);
}

mat3 yzRotate(float theta)
{
    vec2 cs = vec2(cos(theta), sin(theta));
    return mat3(
        1, 0, 0,
        0, cs.x, cs.y,
        0, -cs.y, cs.x);
}

mat3 xzRotate(float theta)
{
    vec2 cs = vec2(cos(theta), sin(theta));
    return mat3(
        cs.x, 0, cs.y,
        0, 1, 0,
        -cs.y, 0, cs.x);
}