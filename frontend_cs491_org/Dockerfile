# Step 1: Build the React application
FROM node:18.12.1 AS build
WORKDIR /app
COPY package*.json ./
COPY . ./
RUN npm install --legacy-peer-deps
# CHECK THIS LATER:
ENV GENERATE_SOURCEMAP=false 
RUN npm run build

# Step 2: Serve the React application with Nginx
FROM nginx:stable-alpine
COPY --from=build /app/build /usr/share/nginx/html

# Overwrite the default Nginx configuration with our custom configuration
COPY nginx/default.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
