// The buffer generates the heightmap of the Mandelbrot set by using smooth iteration
// count with a little bit of orbit trapping to spice things up. Unfortunately I got
// greedy with this whole shader and supersampling is too expensive.

// Lower this for the biggest performance boost
const float maxIters = 400.0;

// This tool is always useful for finding interesting regions (with coordinates)
// within double precision limits
// https://sciencedemos.org.uk/mandelbrot.php
//const vec3 zoomCenter = 0.5 * vec3(
//    -0.111792418416	- 0.105212213568,
//    -0.902231313283 - 0.899972653438, 10.0);
//const vec3 zoomCenter2 = 1.0 * vec3(
//    -0.74592,
//    0.1127,
//    16.0
//);

#define PI 3.1415926538

const float speed = 0.4;
// Points from http://www.cuug.ab.ca/dewara/mandelbrot/Mandelbrowser.html
vec3 coordinates[4] = vec3[4](
	vec3(
		-0.745428,
		0.113009,
		15.0
	),
	vec3(
		-0.748,
		0.1,
		12.0
	),
	vec3(
		-0.1085023160,
		-0.9011019834,
		10.0
	),
	vec3(
		-0.74592,
		0.1127,
		16.0
	)
);
int centerIndex = int(floor(iTime * speed / (2 * PI))) % 4;
dvec2 zoomCenter = dvec2(coordinates[centerIndex].x, coordinates[centerIndex].y);
float zoomMax = coordinates[centerIndex].z;

dvec2 cMul(dvec2 c1, dvec2 c2)
{
    return dvec2(
        c1.x * c2.x - c1.y * c2.y,
        c1.x * c2.y + c1.y * c2.x);
}

// "Triangular" distance from origin
double triangleOrbit(dvec2 point)
{
    double r = length(point);
    double theta = atan(float(point.y), float(point.x));
    theta = mod(theta + pi/3.0, twoPi/3.0) - pi/3.0;
    return r * cos(float(theta));
}

// Returns iteration count and triangle orbit distance
vec2 mandelbrot(dvec2 C)
{
    dvec2 z = C;
    double zz; // z . z
    double orbitDist = 100.0;
    float iters;
    const float bound = 32.0;
    
    // This optimizes the smooth iteration formula and brings it into the format I want
    const float smoothIterOffset = -log2(log2(bound)) - 2.0;
    
    for (iters = 0.0; iters < maxIters; iters++)
    {
        z = cMul(z, z) + C;
        zz = dot(z, z);
        if (zz >= bound * bound)
            break;
        orbitDist = min(orbitDist, triangleOrbit(z));
    }
    // See Inigo Quilez's article on smooth iteration count
    // https://iquilezles.org/www/articles/mset_smooth/mset_smooth.htm
    if (iters != maxIters)
        iters -= (log2(log2(float(zz))) + smoothIterOffset);
        
    return vec2(iters, float(orbitDist));
}


void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    dvec2 dFragCoord = dvec2(fragCoord.x, fragCoord.y);
    dvec2 uv = (2.0 * dFragCoord - iResolution.xy) / min(iResolution.x, iResolution.y);
    
    float smoothTime; // Time that smoothly transitions from 0 to normal speed
    if (iTime < 2.0)
        smoothTime = 0.25 * iTime * iTime;
    else
        smoothTime = iTime - 1.0;
    float rotTime = 0.11 * smoothTime;

    //dvec2 zoomCenter = dvec2(zoomCenter2.x, zoomCenter2.y);

    float zoom = exp2(-zoomMax * (0.5 - 0.5 * cos(speed * iTime)));
    //float zoom = exp2(-zoomMax);

    
    dvec2 center = mix(zoomCenter, dvec2(-0.6, 0.0), pow(zoom, 1.3));
    //dvec2 center = zoomCenter;
    dvec2 point = center + 1.35 * rotate(rotTime) * uv * zoom;
    
    vec2 result = mandelbrot(point);
    float height = pow(log2(float(result.x)) / log2(maxIters), 1.0)
        + 0.05 * cos(3.0 * log2(float(result.y)));
    
    // Half double smallest interval is 1/1024 when the value is between 1 and 2
    // (max height is a little over 1)
    // To get better resolution, the first channel is a quantized height value
    // and the second channel is the fractional component between the steps
    float h1 = floor(height * 1024.0) / 1024.0;
    float h2 = fract(height * 1024.0);
    fragColor = vec4(h1, h2, 1, 1);
}
