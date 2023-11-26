#include <iostream>
#include <fstream>
#include <limits>
#include "raytracer.h"
#include <cmath>
#include "json.hpp"

Vec3 Vec3::operator+(const Vec3& other) const {
    return {x + other.x, y + other.y, z + other.z};
}

Vec3 Vec3::operator-() const {
    return {-x, -y, -z};
}

Vec3& Vec3::operator+=(const Vec3& other) {
    x += other.x;
    y += other.y;
    z += other.z;
    return *this;
}

Vec3 Vec3::operator-(const Vec3& other) const {
    return {x - other.x, y - other.y, z - other.z};
}


Vec3 Vec3::operator*(float scalar) const {
    return {x * scalar, y * scalar, z * scalar};
}

Vec3 Vec3::operator*(const Vec3& other) const {
    return {x * other.x, y * other.y, z * other.z};
}

Vec3 Vec3::operator*(double scalar) const {
    return {x * scalar, y * scalar, z * scalar};
}

Vec3 Vec3::operator/(double scalar) const {
    return {x / scalar, y / scalar, z / scalar};
};


Vec3 Vec3::normalize() const {
    double length = std::sqrt(x * x + y * y + z * z);
    if (length > 0) {
        return {x / length, y / length, z / length};
    } else {
        // Return a default normalized vector with some direction
        return {1.0, 0.0, 0.0}; // or any other non-zero default value
    }
}


float Vec3::dot(const Vec3& other) const {
    return x * other.x + y * other.y + z * other.z;
}

Vec3 Vec3::cross(const Vec3& other) const {
    return {y * other.z - z * other.y, z * other.x - x * other.z, x * other.y - y * other.x};
}



// Function to check if a ray hits the sphere
bool Sphere::intersect(const Ray& ray, float& t, Vec3& hitColor, Vec3& normal) const {
    Vec3 oc = ray.origin - center;
    float a = ray.direction.dot(ray.direction);
    float b = 2.0f * oc.dot(ray.direction);
    float c = oc.dot(oc) - radius * radius;
    float discriminant = b * b - 4 * a * c;

    if (discriminant > 0) {
        float sqrtDiscriminant = std::sqrt(discriminant);
        float t0 = (-b - sqrtDiscriminant) / (2.0f * a);
        float t1 = (-b + sqrtDiscriminant) / (2.0f * a);

        // We only want the closest positive intersection (t > 0).
        if (t0 > 1e-4 && t0 < t1) {
            t = t0;
        } else if (t1 > 1e-4) {
            t = t1;
        } else {
            // Both t0 and t1 are negative which means the sphere is behind the ray origin.
            return false;
        }

        // Calculate normal at the point of intersection
        // The intersection point is calculated using the original ray's direction
        Vec3 intersectionPoint = ray.origin + ray.direction * t;
        normal = (intersectionPoint - center).normalize();

        hitColor = material.diffusecolor;  // Assign material color on hit
        return true;
    }

    return false;
}


const Material& Sphere::getMaterial() const {
    return material;
}

// Function to check if a ray hits the cylinder
bool Cylinder::intersect(const Ray& ray, float& t, Vec3& hitColor, Vec3& normal) const {
    Vec3 CO = ray.origin - center;  // Vector from ray origin to cylinder center
    Vec3 d = ray.direction - axis * ray.direction.dot(axis); // Direction vector projection on the plane perpendicular to the cylinder axis
    Vec3 oc = CO - axis * CO.dot(axis); // Vector from ray origin projected onto the plane perpendicular to the cylinder axis

    // Quadratic coefficients
    float A = d.dot(d);
    float B = 2.0 * oc.dot(d);
    float C = oc.dot(oc) - radius * radius;

    // Discriminant
    float discriminant = B * B - 4 * A * C;
    if (discriminant < 0) {
        return false;  // no intersection
    }

    // The roots of the quadratic equation
    float sqrtDiscriminant = std::sqrt(discriminant);
    float t0 = (-B - sqrtDiscriminant) / (2.0 * A);
    float t1 = (-B + sqrtDiscriminant) / (2.0 * A);

    if (t0 > t1) std::swap(t0, t1);

    // Check if the intersections are within the bounds of the cylinder's height
    float y0 = (ray.origin + ray.direction * t0 - center).dot(axis);
    float y1 = (ray.origin + ray.direction * t1 - center).dot(axis);

    bool intersectsWithinBounds = false;

    // Check if intersection points are within the bounds of the cylinder's height
    if (y0 > 0 && y0 < height) {
        t = t0;
        intersectsWithinBounds = true;
        // Calculate normal at the point of intersection on the cylinder's surface
        Vec3 intersectionPoint = ray.origin + ray.direction * t;
        normal = (intersectionPoint - center - axis * (intersectionPoint - center).dot(axis)).normalize();
        hitColor = material.diffusecolor;  // Assign material color on hit
    } else if (y1 > 0 && y1 < height) {
        t = t1;
        intersectsWithinBounds = true;
        // Calculate normal at the point of intersection on the cylinder's surface
        Vec3 intersectionPoint = ray.origin + ray.direction * t;
        normal = (intersectionPoint - center - axis * (intersectionPoint - center).dot(axis)).normalize();
        hitColor = material.diffusecolor;  // Assign material color on hit
    }

    // If there was no intersection within bounds, check for intersection with the caps
    // Check for intersection with the bottom and top caps
    Vec3 capCenter;
    float tCap;
    Vec3 pCap;  // Intersection point with the cap

    // Checking bottom cap intersection
    capCenter = center;
    if (std::abs(ray.direction.dot(axis)) > 1e-6) {
        tCap = (capCenter - ray.origin).dot(axis) / ray.direction.dot(axis);
        pCap = ray.origin + ray.direction * tCap;
        if ((pCap - capCenter).dot(pCap - capCenter) <= radius * radius && tCap < t) {
            t = tCap;
            intersectsWithinBounds = true;
            normal = -axis;
            hitColor = material.diffusecolor;
        }
    }

    // Checking top cap intersection
    capCenter = center + axis * height;
    if (std::abs(ray.direction.dot(axis)) > 1e-6) {
        tCap = (capCenter - ray.origin).dot(axis) / ray.direction.dot(axis);
        pCap = ray.origin + ray.direction * tCap;
        if ((pCap - capCenter).dot(pCap - capCenter) <= radius * radius && tCap < t) {
            t = tCap;
            intersectsWithinBounds = true;
            normal = axis;
            hitColor = material.diffusecolor;
        }
    }


    return intersectsWithinBounds;
}

const Material& Cylinder::getMaterial() const {
    return material;
}

// Function to check if a ray hits the triangle
bool Triangle::intersect(const Ray& ray, float& t, Vec3& hitColor, Vec3& normal) const {
    const double EPSILON = 1e-6;

    Vec3 edge1 = v1 - v0;
    Vec3 edge2 = v2 - v0;
    Vec3 h = ray.direction.cross(edge2);
    double a = edge1.dot(h);

    if (a > -EPSILON && a < EPSILON) {
        return false; // This ray is parallel to this triangle.
    }

    double f = 1.0 / a;
    Vec3 s = ray.origin - v0;
    double u = f * s.dot(h);

    if (u < 0.0 || u > 1.0) {
        return false; // The intersection is outside of the triangle.
    }

    Vec3 q = s.cross(edge1);
    double v = f * ray.direction.dot(q);

    if (v < 0.0 || u + v > 1.0) {
        return false; // The intersection is outside of the triangle.
    }

    // At this stage we can compute t to find out where the intersection point is on the line.
    double t_temp = f * edge2.dot(q);

    if (t_temp > EPSILON) { // ray intersection
        t = t_temp;

        // Calculate normal for this triangle
        normal = edge1.cross(edge2).normalize();
        if (normal.dot(ray.direction) > 0) { 
            normal = -normal; // The normal should be facing the opposite direction of the ray for the visible side.
        }

        hitColor = material.diffusecolor; // Assign material color on hit
        return true;
    }

    // This means that there is a line intersection but not a ray intersection.
    return false;
}


const Material& Triangle::getMaterial() const {
    return material;
}



// Function to find the closest intersection point in the scene
const Object* Scene::intersect(const Ray& ray, float& t, Vec3& hitColor, Vec3& normal, const Object* originatingObject) const {
    const Object* hitObject = nullptr;  // Initialize hit object as nullptr
    t = std::numeric_limits<float>::infinity();

    for (const auto& object : objects) {
        if (object.get() == originatingObject) {
            // Skip intersection test for the originating object
            continue;
        }
                
        float tempT;
        Vec3 tempHitColor;
        Vec3 tempNormal;
        if (object->intersect(ray, tempT, tempHitColor, tempNormal) && tempT < t) {
            t = tempT;
            hitColor = tempHitColor;
            normal = tempNormal;
            hitObject = object.get();
        }
    }

    return hitObject;
}

const Material& Scene::getIntersectedObjectMaterial(const Ray& ray, float t, Vec3& normal) const {
    const Material* foundMaterial = nullptr;
    float minT = std::numeric_limits<float>::infinity();

    // Default material
    Material defaultMaterial;
    defaultMaterial.ks = 0.1;
    defaultMaterial.kd = 0.1;  
    defaultMaterial.specularexponent = 0;
    defaultMaterial.diffusecolor = {1.0, 1.0, 1.0};
    defaultMaterial.specularcolor = {1.0, 1.0, 1.0};
    defaultMaterial.isreflective = false;
    defaultMaterial.reflectivity = 0.0;
    defaultMaterial.isrefractive = false;
    defaultMaterial.refractiveindex = 0.0;

    // Iterate through objects
    for (const auto& object : objects) {
        float tempT;
        Vec3 tempNormal;
        Vec3 tempHitColor;
        if (object->intersect(ray, tempT, tempHitColor, tempNormal) && tempT < minT && tempT == t) {

            minT = tempT;
            foundMaterial = &object->getMaterial();
            normal = tempNormal;
        }
    }

    // Return the material of the intersected object (or default material if none found)
    return (foundMaterial != nullptr) ? *foundMaterial : defaultMaterial;
}


Vec3 reinhardToneMapping(const Vec3& color, float exposure) {
    Vec3 mappedColor = color * exposure;
    float lum = mappedColor.x * 0.2126 + mappedColor.y * 0.7152 + mappedColor.z * 0.0722;
    mappedColor = mappedColor / (1.0 + lum);
    return mappedColor;
}



Vec3 calculateBlinnPhongShading(const Vec3& intersectionPoint, const Vec3& viewDirection, const Vec3& normal, const Material& material, const Scene& scene, const Object* intersectedObject) {
    Vec3 ambientColor = {0.5, 0.5, 0.5}; // Ambient reflection coefficient
    Vec3 totalColor = ambientColor * material.diffusecolor; // Ambient reflection

    for (const auto& light : scene.lights) {
        Vec3 lightDirection;

        if (light.type == "pointlight") {
            // Point light: Direction from intersection point to light source
            lightDirection = (light.position - intersectionPoint).normalize();
        } else if (light.type == "directionallight") {
            // Directional light: Direction of the light is constant
            lightDirection = (-light.position).normalize();
        } else {
            // Placeholder for unsupported light types
            std::cerr << "Warning: Unsupported light type '" << light.type << "'. Skipping.\n";
            continue; // Skip unsupported light types for now
        }

        // Ensure vectors are normalized
        Vec3 normalizedNormal = normal.normalize();
        Vec3 normalizedLightDirection = lightDirection.normalize();
        Vec3 normalizedHalfwayVector = (viewDirection + normalizedLightDirection).normalize();

        // Check if the point is in shadow by casting a shadow ray to each object in the scene
        Ray shadowRay = {intersectionPoint + normalizedNormal * 0.001, lightDirection};
        float shadowT;
        Vec3 shadowHitColor, shadowNormal;

        // Check for intersections with other objects in the scene
        bool inShadow = false;

        // Pass the intersected object to avoid self-shadowing
        if (scene.intersect(shadowRay, shadowT, shadowHitColor, shadowNormal, intersectedObject) && shadowT > 0.001) {
            inShadow = true;
        }

        for (const auto& object : scene.objects) {
            if (object->intersect(shadowRay, shadowT, shadowHitColor, shadowNormal) && shadowT > 0.001) {
                // Point is in shadow, skip diffuse and specular calculations
                inShadow = true;
                break;
            }
        }

        if (inShadow) {
            continue;
        }

        // Diffuse reflection
        float diffuseFactor = std::max(0.0f, normalizedLightDirection.dot(normalizedNormal));
        Vec3 diffuseColor = light.intensity * material.diffusecolor * diffuseFactor * material.kd;

        // Specular reflection (Blinn-Phong model)
        float specularFactor = std::pow(std::max(0.0f, normalizedNormal.dot(normalizedHalfwayVector)), material.specularexponent);
        Vec3 specularColor = light.intensity * material.specularcolor * specularFactor * material.ks;

        totalColor += diffuseColor + specularColor;
    }

    return totalColor;
}


Vec3 traceRay(const Ray& ray, const Scene& scene, int maxBounces, const Object* originatingObject, int currentBounce) {
    if (maxBounces <= 0) {
        return scene.backgroundColor; // Use scene background color for reflections that reach max bounces
    }

    float t;
    Vec3 hitColor;
    Vec3 normal;

    const Object* intersectedObject = scene.intersect(ray, t, hitColor, normal, originatingObject);

    if (intersectedObject && t > 1e-4) {
        Vec3 intersectionPoint = ray.origin + ray.direction * t;
        Material intersectedMaterial = scene.getIntersectedObjectMaterial(ray, t, normal);
        Vec3 shadingResult = calculateBlinnPhongShading(intersectionPoint, -ray.direction, normal, intersectedMaterial, scene, intersectedObject);

        // Recursive reflection (if the material is reflective)
        // Recursive reflection (if the material is reflective)
        if (intersectedMaterial.isreflective && maxBounces > 0) {
            Vec3 normalizedRayDirection = ray.direction.normalize();
            Vec3 normalizedNormal = normal.normalize();

            // Check the dot product to ensure we're only reflecting off the front side of the sphere
            if (normalizedNormal.dot(normalizedRayDirection) < 0) {
                Vec3 reflectionDirection = normalizedRayDirection - (normalizedNormal * 2.0f * normalizedRayDirection.dot(normalizedNormal));
                reflectionDirection = reflectionDirection.normalize(); // Normalize the reflection direction

                // Offset the origin of the reflection ray slightly to avoid self-intersection
                Vec3 reflectedRayOrigin = intersectionPoint + normalizedNormal * 1e-4;

                Ray reflectedRay = {reflectedRayOrigin, reflectionDirection};

                // Trace the reflection ray
                Vec3 reflectionColor = traceRay(reflectedRay, scene, maxBounces - 1, intersectedObject, currentBounce + 1) * intersectedMaterial.reflectivity;

                hitColor = reflectionColor + (shadingResult * (1.0 - intersectedMaterial.reflectivity));
            } else {
                // If the normal and the ray direction are facing the same direction, don't calculate reflection
                hitColor = shadingResult;
            }
        } else {
            // No reflection, use direct shading result
            hitColor = shadingResult;
        }

        // Recursive refraction (if the material is refractive)
        if (intersectedMaterial.isrefractive && maxBounces > 0) {
            float n1 = 1.0f; // Refractive index of air
            float n2 = intersectedMaterial.refractiveindex; // Refractive index of the object

            Vec3 normalizedRayDirection = ray.direction.normalize();
            Vec3 normalizedNormal = normal.normalize();

            float cosI = -normalizedNormal.dot(normalizedRayDirection);
            if (cosI < 0) { // Ray is inside the object
                cosI = -cosI;
                normalizedNormal = -normalizedNormal;
                std::swap(n1, n2);
            }

            float eta = n1 / n2;
            float k = 1 - eta * eta * (1 - cosI * cosI);
            if (k < 0) { 
                // Total internal reflection

                Vec3 reflectionDirection = normalizedRayDirection - (normalizedNormal * 2.0f * normalizedRayDirection.dot(normalizedNormal));
                reflectionDirection = reflectionDirection.normalize(); // Normalize the reflection direction

                // Add a small offset to the intersection point to avoid self-intersection
                Vec3 offset = normalizedNormal * 1e-4;
                Vec3 reflectedRayOrigin = intersectionPoint + offset;

                Ray reflectedRay = {reflectedRayOrigin, reflectionDirection};

                // Use the traceRay function recursively for the reflected ray
                Vec3 totalInternalReflectionColor = traceRay(reflectedRay, scene, maxBounces - 1, intersectedObject, currentBounce + 1);

                // Return the color resulting from total internal reflection
                return totalInternalReflectionColor;
            } else {
                Vec3 refractionDirection = (normalizedRayDirection * eta) + (normalizedNormal * (eta * cosI - sqrt(k)));
                refractionDirection = refractionDirection.normalize();  // Normalize the refraction direction
                // Add a small offset to the intersection point to avoid self-intersection
                Vec3 offset = -normalizedNormal * 1e-4;
                Vec3 refractedRayOrigin = intersectionPoint + offset;

                Ray refractedRay = {refractedRayOrigin, refractionDirection};

                intersectedMaterial.transparency = 0.5;

                Vec3 refractionColor = traceRay(refractedRay, scene, maxBounces - 1, intersectedObject, currentBounce + 1) * intersectedMaterial.transparency;

                // Blend refractionColor with shadingResult
                hitColor = refractionColor + (shadingResult * (1.0 - intersectedMaterial.transparency));
            }
        }

        // Clamp the resulting color to avoid values greater than 1.0
        hitColor.x = std::min(hitColor.x, 1.0);
        hitColor.y = std::min(hitColor.y, 1.0);
        hitColor.z = std::min(hitColor.z, 1.0);

        return hitColor;
    }

    // Return scene background color only if the material is not reflective
    return scene.backgroundColor;
}


void writePPMHeader(std::ofstream& ppmFile, int width, int height) {
    ppmFile << "P3\n" << width << " " << height << "\n255\n";
}

void writeColor(std::ofstream& ppmFile, const Vec3& color) {
    ppmFile << static_cast<int>(color.x * 255) << " "
            << static_cast<int>(color.y * 255) << " "
            << static_cast<int>(color.z * 255) << " ";
}

int main() {
    try {
        // Read the JSON file
        std::ifstream file("scene.json");
        if (!file.is_open()) {
            std::cerr << "Error opening file." << std::endl;
            return 1;
        }

        // Parse the JSON
        json sceneData;
        file >> sceneData;

        //Extract render mode
        std::string renderMode = sceneData["rendermode"];

        // Extract camera information
        Camera camera;
        camera.type = sceneData["camera"]["type"];
        camera.width = sceneData["camera"]["width"];
        camera.height = sceneData["camera"]["height"];
        camera.position = {sceneData["camera"]["position"][0], sceneData["camera"]["position"][1], sceneData["camera"]["position"][2]};
        camera.lookAt = {sceneData["camera"]["lookAt"][0], sceneData["camera"]["lookAt"][1], sceneData["camera"]["lookAt"][2]};
        camera.upVector = {sceneData["camera"]["upVector"][0], sceneData["camera"]["upVector"][1], sceneData["camera"]["upVector"][2]};
        camera.fov = sceneData["camera"]["fov"];
        camera.exposure = sceneData["camera"]["exposure"];


        // Extract scene information
        Scene scene;
        scene.backgroundColor = {sceneData["scene"]["backgroundcolor"][0], sceneData["scene"]["backgroundcolor"][1], sceneData["scene"]["backgroundcolor"][2]};

        for (const auto& shape : sceneData["scene"]["shapes"]) {
            if (shape["type"] == "sphere") {
                std::unique_ptr<Sphere> sphere = std::make_unique<Sphere>();
                sphere->center = {shape["center"][0], shape["center"][1], shape["center"][2]};
                sphere->radius = shape["radius"];
                if (renderMode == "phong") {
                    sphere->material.ks = shape["material"]["ks"];
                    sphere->material.kd = shape["material"]["kd"];
                    sphere->material.specularexponent = shape["material"]["specularexponent"];
                    sphere->material.diffusecolor = {
                        shape["material"]["diffusecolor"][0],
                        shape["material"]["diffusecolor"][1],
                        shape["material"]["diffusecolor"][2]
                    };
                    sphere->material.specularcolor = {
                        shape["material"]["specularcolor"][0],
                        shape["material"]["specularcolor"][1],
                        shape["material"]["specularcolor"][2]
                    };
                    sphere->material.isreflective = shape["material"]["isreflective"];
                    sphere->material.reflectivity = shape["material"]["reflectivity"];
                    sphere->material.isrefractive = shape["material"]["isrefractive"];
                    sphere->material.refractiveindex = shape["material"]["refractiveindex"];
                }
                scene.objects.push_back(std::move(sphere));

            } else if (shape["type"] == "cylinder") {
                std::unique_ptr<Cylinder> cylinder = std::make_unique<Cylinder>();
                cylinder->center = {shape["center"][0], shape["center"][1], shape["center"][2]};
                cylinder->axis = {shape["axis"][0], shape["axis"][1], shape["axis"][2]};
                cylinder->radius = shape["radius"];
                cylinder->height = shape["height"];
                if (renderMode == "phong"){
                    cylinder->material.ks = shape["material"]["ks"];
                    cylinder->material.kd = shape["material"]["kd"];
                    cylinder->material.specularexponent = shape["material"]["specularexponent"];
                    cylinder->material.diffusecolor = {
                        shape["material"]["diffusecolor"][0],
                        shape["material"]["diffusecolor"][1],
                        shape["material"]["diffusecolor"][2]
                    };
                    cylinder->material.specularcolor = {
                        shape["material"]["specularcolor"][0],
                        shape["material"]["specularcolor"][1],
                        shape["material"]["specularcolor"][2]
                    };
                    cylinder->material.isreflective = shape["material"]["isreflective"];
                    cylinder->material.reflectivity = shape["material"]["reflectivity"];
                    cylinder->material.isrefractive = shape["material"]["isrefractive"];
                    cylinder->material.refractiveindex = shape["material"]["refractiveindex"];
                }
                scene.objects.push_back(std::move(cylinder));
            } else if (shape["type"] == "triangle") {
                std::unique_ptr<Triangle> triangle = std::make_unique<Triangle>();
                triangle->v0 = {shape["v0"][0], shape["v0"][1], shape["v0"][2]};
                triangle->v1 = {shape["v1"][0], shape["v1"][1], shape["v1"][2]};
                triangle->v2 = {shape["v2"][0], shape["v2"][1], shape["v2"][2]};
                if (renderMode == "phong"){
                    triangle->material.ks = shape["material"]["ks"];
                    triangle->material.kd = shape["material"]["kd"];
                    triangle->material.specularexponent = shape["material"]["specularexponent"];
                    triangle->material.diffusecolor = {
                        shape["material"]["diffusecolor"][0],
                        shape["material"]["diffusecolor"][1],
                        shape["material"]["diffusecolor"][2]
                    };
                    triangle->material.specularcolor = {
                        shape["material"]["specularcolor"][0],
                        shape["material"]["specularcolor"][1],
                        shape["material"]["specularcolor"][2]
                    };
                    triangle->material.isreflective = shape["material"]["isreflective"];
                    triangle->material.reflectivity = shape["material"]["reflectivity"];
                    triangle->material.isrefractive = shape["material"]["isrefractive"];
                    triangle->material.refractiveindex = shape["material"]["refractiveindex"];
                }
                scene.objects.push_back(std::move(triangle));
            }
        }

        // Extract light information
        if (renderMode == "phong"){
            for (const auto& lightData : sceneData["scene"]["lightsources"]) {
                Light light;
                light.type = lightData["type"];
                light.position = {lightData["position"][0], lightData["position"][1], lightData["position"][2]};
                light.intensity = {lightData["intensity"][0], lightData["intensity"][1], lightData["intensity"][2]};
                scene.lights.push_back(light);
            }
        }



        // Create an image buffer (placeholder, replace with your rendering logic)
        std::vector<std::vector<Vec3>> image(camera.height, std::vector<Vec3>(camera.width, {0.0, 0.0, 0.0}));

        // Output the PPM image
        std::ofstream ppmFile("output_image.ppm");
        if (!ppmFile.is_open()) {
            std::cerr << "Error creating PPM file." << std::endl;
            return 1;
        }

        writePPMHeader(ppmFile, camera.width, camera.height);

        // Write pixel values
        
        for (int i = camera.height - 1; i >= 0; --i) {
            for (int j = 0; j < camera.width; ++j) {
                Ray ray;
                // Compute camera coordinate system
                Vec3 forward = (camera.lookAt - camera.position).normalize();
                Vec3 right = camera.upVector.normalize().cross(forward).normalize();
                Vec3 up = forward.cross(right).normalize();

                // Compute ray direction in screen space with FOV
                #define M_PI 3.14159265358979323846264338327950288
                float aspectRatio = static_cast<float>(camera.width) / static_cast<float>(camera.height);
                float screenX = (2.0f * (static_cast<float>(j) + 0.5f) / static_cast<float>(camera.width) - 1.0f) * aspectRatio * std::tan((camera.fov * M_PI / 180.0) / 2.0);
                float screenY = (2.0f * (static_cast<float>(i) + 0.5f) / static_cast<float>(camera.height) - 1.0f) * std::tan((camera.fov * M_PI / 180.0) / 2.0);

                // Transform ray direction to world space
                ray.direction = (right * screenX + up * screenY + forward).normalize();

                // Set ray origin to camera position
                ray.origin = camera.position;

                if (renderMode == "phong"){                  
                // Extract 'nbounces' parameter
                    scene.maxBounces = sceneData["nbounces"];
                }
                else{
                    scene.maxBounces = 1;
                }

                int maxBounces = scene.maxBounces;
                Vec3 color = traceRay(ray, scene, maxBounces);


                color = reinhardToneMapping(color, 1.5); // Apply tone mapping
                writeColor(ppmFile, color);
            }
            ppmFile << "\n";
        }
        ppmFile.close();
    } catch (const json::exception& e) {
        std::cerr << "JSON parsing error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}