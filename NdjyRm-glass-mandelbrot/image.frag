// Looking through glass with the Mandelbrot set imprinted. There's a few
// approximations/corners cut to simplify things but it still looks good.

// Performance boosters in order of importance: reduce maxIters in buffer A,
// disable DISPERSE, disable FRESNEL.

// Buffer A renders the heightmap, which is used here to calculate the surface normal
// and have a field day from there.

// I sincerely apologize for how messy the preprocessor directives are
// Created by Anthony Hall


// Refracts an individual beam for each color channel
#define DISPERSE

// Introduces some reflection according to (slightly modified) Fresnel equation
#define FRESNEL

// Index must be at least 1.0 as the shader is not designed to handle total internal
// reflection. 1.52 is the refractive index of glass, disperse intensity is arbitrary
const float refractiveIndex = 1.52;
const float disperseIntensity = 0.02;

// This is inversely proportional to the max height of the heightmap, i.e. raising this
// will treat the map as flatter
const float normalZScale = 5.0;

const float fov = radians(70.0);

float convertHeight(vec2 rawHeight)
{
    return rawHeight.x + rawHeight.y / 1024.0;
}

// This treats the top and right most pixels as flat, but it's not very noticeable
// so I chose not to spend extra computation on handling that case better
vec3 getNormal(vec2 fragCoord, float height)
{
    fragCoord = min(fragCoord, iResolution.xy - 2.0);
    float heightX = convertHeight(texelFetch(iChannel0, ivec2(fragCoord + vec2(0, 1)), 0).xy);
    float heightY = convertHeight(texelFetch(iChannel0, ivec2(fragCoord + vec2(1, 0)), 0).xy);
    
    float dx = heightX - height;
    float dy = heightY - height;
    return normalize(vec3(-dx, -dy, normalZScale / min(iResolution.x, iResolution.y))); // Correct against resolution
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    // The ray intersection is treated as if the glass is flat because doing
    // otherwise is more expensive than it's worth
    vec2 uv = (2.0 * fragCoord - iResolution.xy) / min(iResolution.x, iResolution.y);
    float height = convertHeight(texelFetch(iChannel0, ivec2(fragCoord), 0).xy);
    vec3 normal = getNormal(fragCoord, height);
    
    float effectMix = smoothstep(0.0, 3.0, iTime); // Strength of refraction/dispersion
    float index = 1.0 + (refractiveIndex - 1.0) * effectMix;
    
    vec3 rayDir = normalize(vec3(uv * tan(fov / 2.0), -1.0));
    vec3 refracted = refract(rayDir, normal, 1.0 / index);
    
    // Rotation/mouse input
    float xyTheta;
    float yzTheta;
    float xzTheta;
    if (iMouse.z > 0.0)
    {
        xyTheta = 0.0;
        yzTheta = (iMouse.y / iResolution.y - 0.5) * pi;
        xzTheta = (2.0 * iMouse.x / iResolution.x - 1.0) * pi;
    }
    else
    {
        xyTheta = -0.031 * iTime;
        yzTheta = 0.45 - 0.05 * cos(0.11 * iTime);
        xzTheta = -0.04 * iTime;
    }
    mat3 rayMat = xyRotate(xyTheta);
    rayMat = yzRotate(yzTheta) * rayMat;
    rayMat = xzRotate(xzTheta) * rayMat;
    
#ifdef DISPERSE
    vec3 refractedG = refract(rayDir, normal,
        1.0 / (index + effectMix * disperseIntensity));
    vec3 refractedB = refract(rayDir, normal,
        1.0 / (index + 2.0 * effectMix * disperseIntensity));
#endif
    
#ifdef FRESNEL
    // Fresnel for S polarization - It's already subtle, so it's hardly noticeable at
    // all with any amount of P polarization reflectance factored in. Since it's subtle
    // I also change the exponent to bring up the low values a bit
    float cosIncident = abs(dot(rayDir, normal));
    float cosRefracted = abs(dot(refracted, normal));
    float reflectance = (cosIncident / index - cosRefracted)
        / (cosIncident / index + cosRefracted);
    reflectance = pow(abs(reflectance), 1.7); // 2.0 for scientific accuracy
#endif
    
    refracted = rayMat * refracted;
    
#ifdef DISPERSE
    refractedG = rayMat * refractedG;
    refractedB = rayMat * refractedB;
#endif
    
#ifdef FRESNEL

#ifdef DISPERSE
    vec3 transmitColor = vec3(
        texture(iChannel1, refracted).r,
        texture(iChannel1, refractedG).g,
        texture(iChannel1, refractedB).b);
#else
    vec3 transmitColor = texture(iChannel1, refracted).rgb;
#endif

    vec3 reflected = reflect(refracted, normal);
    vec3 reflectColor = texture(iChannel1, reflected).rgb;
    vec3 color = mix(transmitColor, reflectColor, reflectance);
    
#else

#ifdef DISPERSE
    vec3 color = vec3(
        texture(iChannel1, refracted).r,
        texture(iChannel1, refractedG).g,
        texture(iChannel1, refractedB).b);
#else
    vec3 color = texture(iChannel1, refracted).rgb;
#endif

#endif

    fragColor = vec4(color, 1.0);
}