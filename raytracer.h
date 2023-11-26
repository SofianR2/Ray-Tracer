#ifndef RAYTRACER_H
#define RAYTRACER_H

#include <vector>
#include <string>
#include <fstream>
#include "json.hpp"

using json = nlohmann::json;

struct Vec3 {
    double x, y, z;

    Vec3 operator+(const Vec3& other) const;
    Vec3 operator-() const;
    Vec3& operator+=(const Vec3& other);
    Vec3 operator-(const Vec3& other) const;
    Vec3 operator*(float scalar) const;
    Vec3 operator*(const Vec3& other) const;
    Vec3 operator*(double scalar) const;
    Vec3 operator/(double scalar) const;
    Vec3 normalize() const;
    float dot(const Vec3& other) const;
    Vec3 cross(const Vec3& other) const;
};

struct Ray {
    Vec3 origin;
    Vec3 direction;
};

struct Camera {
    std::string type;
    int width, height;
    Vec3 position, lookAt, upVector;
    float fov, exposure;
};

struct Material {
    float ks, kd;
    int specularexponent;
    Vec3 diffusecolor, specularcolor;
    bool isreflective;
    float reflectivity;
    bool isrefractive;
    float refractiveindex;
    float transparency;
};

struct Object {
    virtual bool intersect(const Ray& ray, float& t, Vec3& hitColor, Vec3& normal) const = 0;
    virtual const Material& getMaterial() const = 0;
    virtual ~Object() = default;
};

struct Sphere : public Object {
    Vec3 center;
    float radius;
    Material material;

    bool intersect(const Ray& ray, float& t, Vec3& hitColor, Vec3& normal) const override;
    const Material& getMaterial() const override;
};

struct Cylinder : public Object {
    Vec3 center, axis;
    float radius, height;
    Material material;

    bool intersect(const Ray& ray, float& t, Vec3& hitColor, Vec3& normal) const override;
    const Material& getMaterial() const override;
};

struct Triangle : public Object {
    Vec3 v0, v1, v2;
    Material material;

    bool intersect(const Ray& ray, float& t, Vec3& hitColor, Vec3& normal) const override;
    const Material& getMaterial() const override;
};

struct Light {
    std::string type;
    Vec3 position;
    Vec3 intensity;
};

struct Scene {
    Vec3 backgroundColor;
    std::vector<std::unique_ptr<Object>> objects;
    std::vector<Light> lights;
    int maxBounces;

    const Object* intersect(const Ray& ray, float& t, Vec3& hitColor, Vec3& normal, const Object* originatingObject = nullptr) const;
    const Material& getIntersectedObjectMaterial(const Ray& ray, float t, Vec3& normal) const;
};

Vec3 reinhardToneMapping(const Vec3& color, float exposure);
Vec3 calculateBlinnPhongShading(const Vec3& intersectionPoint, const Vec3& viewDirection, const Vec3& normal, const Material& material, const Scene& scene, const Object* intersectedObject);
Vec3 traceRay(const Ray& ray, const Scene& scene, int maxBounces, const Object* originatingObject = nullptr, int currentBounce = 0);
void writePPMHeader(std::ofstream& ppmFile, int width, int height);
void writeColor(std::ofstream& ppmFile, const Vec3& color);

#endif // RAYTRACER_H
