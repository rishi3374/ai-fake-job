# Deployment Guide

This guide provides step-by-step instructions for deploying the AI Fake Job Detector to web hosting services.

## Deployment Options

### **Option 1: Render.com (Recommended - Free Tier)**

**Services:**
- Backend: Render Web Service
- Frontend: Render Static Site
- Database: Render PostgreSQL

**Setup Steps:**

1. **Create Render Account**
   - Go to [render.com](https://render.com)
   - Sign up with GitHub

2. **Prepare Backend for Deployment**
   
   Create `render.yaml` in project root:
   ```yaml
   services:
     - type: web
       name: ai-fake-job-detector-api
       runtime: python
       buildCommand: pip install -r requirements.txt
       startCommand: uvicorn backend.api.main:app --host 0.0.0.0 --port $PORT
       envVars:
         - key: DATABASE_URL
           fromDatabase:
             name: fake-job-detector-db
             property: connectionString
         - key: PYTHON_VERSION
           value: 3.12.0
   databases:
     - name: fake-job-detector-db
       databaseName: fake_job_detector
       user: fake_job_user
   ```

3. **Deploy Backend**
   - Push code to GitHub
   - Connect GitHub repository to Render
   - Select `render.yaml` configuration
   - Deploy

4. **Deploy Frontend**
   - Create new Static Site on Render
   - Connect to GitHub repository
   - Build command: `cd frontend && npm install && npm run build`
   - Publish directory: `frontend/dist`
   - Deploy

5. **Update Frontend Environment**
   - Add environment variable: `VITE_API_URL=https://your-api-url.onrender.com`

### **Option 2: Railway.app (Free Tier Available)**

**Setup Steps:**

1. **Create Railway Account**
   - Go to [railway.app](https://railway.app)
   - Sign up with GitHub

2. **Deploy from GitHub**
   - Click "New Project" → "Deploy from GitHub repo"
   - Select your repository
   - Railway will automatically detect services

3. **Configure Services**
   
   **Backend Service:**
   - Runtime: Python
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn backend.api.main:app --host 0.0.0.0 --port $PORT`
   
   **Frontend Service:**
   - Runtime: Node
   - Build Command: `cd frontend && npm install && npm run build`
   - Start Command: `cd frontend && npm run preview`
   
   **Database:**
   - Add PostgreSQL database
   - Copy connection string to backend environment variables

4. **Environment Variables**
   ```
   DATABASE_URL=postgresql://...
   API_HOST=0.0.0.0
   API_PORT=$PORT
   SECRET_KEY=your-secret-key
   ```

### **Option 3: PythonAnywhere (Free Tier)**

**Setup Steps:**

1. **Create PythonAnywhere Account**
   - Go to [pythonanywhere.com](https://www.pythonanywhere.com)
   - Sign up for free account

2. **Upload Code**
   - Use git to clone your repository
   - Or upload files via web interface

3. **Configure Web App**
   - Create new Web App
   - Python version: 3.12
   - Framework: Flask (works with FastAPI)
   - WSGI file: `backend/api/main.py`

4. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Configure Environment Variables**
   - Add DATABASE_URL
   - Add SECRET_KEY
   - Add other required variables

### **Option 4: Streamlit (Simplest for Demo)**

**Setup Steps:**

1. **Create Streamlit App**
   
   Create `streamlit_app.py`:
   ```python
   import streamlit as st
   import sys
   from pathlib import Path
   
   sys.path.append(str(Path(__file__).parent.parent))
   
   from models.hybrid.hybrid_model import HybridModel
   
   st.title("AI Fake Job Detector")
   st.write("Analyze job postings for fraud detection")
   
   # Initialize model
   @st.cache_resource
   def load_model():
       return HybridModel()
   
   model = load_model()
   
   # Input fields
   job_description = st.text_area("Job Description")
   company_name = st.text_input("Company Name")
   salary = st.text_input("Salary")
   
   if st.button("Analyze"):
       result = model.predict(
           text=job_description,
           company_data={'name': company_name},
           salary_str=salary
       )
       
       st.write(f"Prediction: {'FAKE' if result['is_fake'] else 'REAL'}")
       st.write(f"Confidence: {result['confidence']:.2%}")
       st.write(f"Risk Level: {result['risk_level']}")
   ```

2. **Deploy to Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect to GitHub repository
   - Select `streamlit_app.py`
   - Deploy

## Pre-Deployment Checklist

### **Backend Preparation**
- [ ] Update `requirements.txt` with all dependencies
- [ ] Set `DEBUG=False` in production
- [ ] Use production database (PostgreSQL)
- [ ] Set strong `SECRET_KEY`
- [ ] Configure CORS for production domain
- [ ] Add proper error handling
- [ ] Implement rate limiting
- [ ] Add logging configuration

### **Frontend Preparation**
- [ ] Update API URL for production
- [ ] Build production bundle
- [ ] Test build locally
- [ ] Optimize images and assets
- [ ] Set environment variables
- [ ] Test on mobile devices

### **Database Preparation**
- [ ] Use PostgreSQL instead of SQLite
- [ ] Set up database backups
- [ ] Configure connection pooling
- [ ] Run migrations if needed
- [ ] Test database connectivity

### **Security**
- [ ] Enable HTTPS
- [ ] Use environment variables for secrets
- [ ] Implement authentication (if needed)
- [ ] Add security headers
- [ ] Regular dependency updates

## Quick Deployment to Render

### **Step 1: Prepare Repository**

```bash
# Ensure all files are committed
git add .
git commit -m "Ready for deployment"

# Push to GitHub
git push origin main
```

### **Step 2: Deploy Backend**

1. Go to [render.com](https://render.com)
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Configure:
   - **Name**: `ai-fake-job-detector-api`
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn backend.api.main:app --host 0.0.0.0 --port $PORT`
5. Add environment variables:
   - `DATABASE_URL`: Add PostgreSQL database
   - `SECRET_KEY`: Generate random key
   - `DEBUG`: `False`
6. Click "Deploy Web Service"

### **Step 3: Deploy Database**

1. Click "New +" → "PostgreSQL"
2. **Name**: `fake-job-detector-db`
3. **Database**: `fake_job_detector`
4. **User**: `fake_job_user`
5. Click "Create Database"
6. Copy connection string to backend environment variables

### **Step 4: Deploy Frontend**

1. Click "New +" → "Static Site"
2. Connect your GitHub repository
3. Configure:
   - **Name**: `ai-fake-job-detector-frontend`
   - **Build Command**: `cd frontend && npm install && npm run build`
   - **Publish Directory**: `frontend/dist`
4. Add environment variable:
   - `VITE_API_URL`: Your backend URL (e.g., `https://ai-fake-job-detector-api.onrender.com`)
5. Click "Deploy Static Site"

### **Step 5: Test Deployment**

1. Wait for both services to be live
2. Open frontend URL in browser
3. Test job analysis functionality
4. Check backend logs for errors

## Troubleshooting

### **Backend Issues**

**Issue: Build fails**
- Check `requirements.txt` for correct versions
- Ensure all dependencies are compatible
- Check build logs for specific errors

**Issue: Runtime errors**
- Check environment variables
- Verify database connection
- Review application logs

**Issue: Database connection**
- Ensure DATABASE_URL is correct
- Check database is running
- Verify network connectivity

### **Frontend Issues**

**Issue: Build fails**
- Check `package.json` dependencies
- Ensure Node version compatibility
- Review build logs

**Issue: API connection errors**
- Verify VITE_API_URL is correct
- Check CORS configuration
- Ensure backend is running

**Issue: Static assets not loading**
- Check publish directory configuration
- Verify build output structure
- Review CDN configuration

## Monitoring and Maintenance

### **Logging**
- Monitor application logs
- Set up error tracking (Sentry, Rollbar)
- Review performance metrics

### **Backups**
- Regular database backups
- Backup model files
- Version control for code

### **Updates**
- Regular dependency updates
- Security patches
- Feature improvements

## Cost Estimation

### **Render.com (Free Tier)**
- Backend: Free (750 hours/month)
- Frontend: Free (100 GB bandwidth/month)
- Database: Free (90 days, then $7/month)
- **Total**: Free for 90 days, then ~$7/month

### **Railway.app**
- Free tier: $5 credit/month
- After credit: ~$5-10/month depending on usage

### **PythonAnywhere**
- Free tier available
- Paid plans start at $5/month

### **Streamlit Cloud**
- Free tier available
- Paid plans start at $10/month

## Performance Optimization

### **Backend**
- Use caching for expensive operations
- Implement rate limiting
- Optimize database queries
- Use connection pooling

### **Frontend**
- Code splitting
- Lazy loading
- Image optimization
- CDN for static assets

### **Database**
- Index optimization
- Query optimization
- Connection pooling
- Regular maintenance

## Scaling Considerations

### **When to Scale**
- High traffic volume
- Slow response times
- Resource constraints
- Growing user base

### **Scaling Options**
- Vertical scaling (more resources)
- Horizontal scaling (multiple instances)
- Load balancing
- CDN integration

## Domain Configuration

### **Custom Domain**
1. Purchase domain from registrar
2. Add domain to hosting service
3. Configure DNS records
4. Enable SSL certificate
5. Update environment variables

## Security Best Practices

### **Production Security**
- Use HTTPS only
- Implement authentication
- Rate limiting
- Input validation
- Regular security audits
- Dependency updates
- Environment variable management

## Support and Maintenance

### **Regular Tasks**
- Monitor application health
- Review logs and metrics
- Update dependencies
- Security patches
- Performance optimization
- User feedback review

### **Emergency Procedures**
- Backup restoration
- Rollback procedures
- Incident response
- Communication plan

## Next Steps

After successful deployment:
1. Monitor performance metrics
2. Gather user feedback
3. Plan improvements
4. Scale as needed
5. Maintain security updates

For additional support, refer to the hosting service documentation or community forums.
