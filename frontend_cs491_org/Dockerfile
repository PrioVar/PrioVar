# Step 1: Build the React application
FROM node:14 AS build
WORKDIR /app
COPY package*.json ./
COPY . ./
RUN npm install
RUN npm run build

# Step 2: Serve the React application with Nginx
FROM nginx:stable-alpine
COPY --from=build /app/build /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
